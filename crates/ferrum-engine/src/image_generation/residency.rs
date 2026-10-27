//! Bounded model session/snapshot slots. Compatible plans share the same sealed
//! physical weights; dynamic storage remains independently admitted.
use super::*;
use ferrum_quantization::bf16_whole_tensor_fp8::ExactBf16TensorSource;
use ferrum_quantization::ManagedDenseArchiveSource;
use std::num::NonZeroU64;

mod adaptive;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ImageResidencyPolicy {
    #[default]
    Cold,
    /// Keep reusable sessions and static weights until a later compiled stage
    /// needs their capacity. Idle dynamic backing is reclaimed before weights.
    /// Actual device-account admission remains authoritative.
    Adaptive,
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
    /// Also retain weightless stage sessions. Each invocation still consumes
    /// fresh request inputs; dynamic backing follows its separate policy.
    AllStages,
    /// Retain only the explicitly selected stage sessions and their readonly
    /// weights. All original capacity and retirement checks still apply.
    SelectedStages { stages: ImageResidentStageSet },
}

impl ImageResidencyPolicy {
    pub fn validate(self) -> anyhow::Result<()> {
        if let Self::SelectedStages { stages } = self {
            ensure!(
                RETAINABLE_STAGES
                    .into_iter()
                    .any(|stage| stages.contains(stage)),
                "selected stage residency requires at least one stage"
            );
        }
        Ok(())
    }

    fn retains(self, stage: ImageStage) -> bool {
        match self {
            Self::Cold => false,
            Self::Adaptive => true,
            Self::DenoiserAndDecoder => {
                matches!(stage, ImageStage::Denoiser | ImageStage::Decoder)
            }
            Self::AllModelComponents => !matches!(stage, ImageStage::Noise),
            Self::AllStages => true,
            Self::SelectedStages { stages } => stages.contains(stage),
        }
    }
}

/// Bounded stage selection. Omitted stages remain cold; selecting a stage does
/// not change its dynamic retention policy or grant any additional capacity.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ImageResidentStageSet {
    #[serde(skip_serializing_if = "is_false")]
    pub text_encoder: bool,
    #[serde(skip_serializing_if = "is_false")]
    pub vision_encoder: bool,
    #[serde(skip_serializing_if = "is_false")]
    pub vae_encoder: bool,
    #[serde(skip_serializing_if = "is_false")]
    pub denoiser: bool,
    #[serde(skip_serializing_if = "is_false")]
    pub decoder: bool,
    #[serde(skip_serializing_if = "is_false")]
    pub noise: bool,
}

impl ImageResidentStageSet {
    fn contains(self, stage: ImageStage) -> bool {
        match stage {
            ImageStage::TextEncoder => self.text_encoder,
            ImageStage::VisionEncoder => self.vision_encoder,
            ImageStage::VaeEncoder => self.vae_encoder,
            ImageStage::Denoiser => self.denoiser,
            ImageStage::Decoder => self.decoder,
            ImageStage::Noise => self.noise,
        }
    }
}

fn is_false(value: &bool) -> bool {
    !*value
}

/// Explicit ceilings for stages that may retain existing idle chunks. Omitted
/// stages release their dynamic backing. This fixed stage vocabulary keeps
/// policy metadata bounded and rejects misspelled or unsupported stages.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ImageStageRetentionBudgets {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_encoder: Option<NonZeroU64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vision_encoder: Option<NonZeroU64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vae_encoder: Option<NonZeroU64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub denoiser: Option<NonZeroU64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoder: Option<NonZeroU64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub noise: Option<NonZeroU64>,
}
impl ImageStageRetentionBudgets {
    pub fn maximum_bytes(self, stage: ImageStage) -> u64 {
        match stage {
            ImageStage::TextEncoder => self.text_encoder,
            ImageStage::VisionEncoder => self.vision_encoder,
            ImageStage::VaeEncoder => self.vae_encoder,
            ImageStage::Denoiser => self.denoiser,
            ImageStage::Decoder => self.decoder,
            ImageStage::Noise => self.noise,
        }
        .map_or(0, NonZeroU64::get)
    }
}

/// Controls idle dynamic backing independently of static model residency.
/// Retention never reserves extra capacity or grants another root reclamation
/// authority. The selected ceiling is included in cross-stage preflight peaks.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ImageDynamicResidencyPolicy {
    #[default]
    Release,
    /// Keep existing idle chunks without reserving additional storage. Requires
    /// adaptive session residency, which can reclaim these chunks on demand.
    Adaptive,
    RetainDecoder {
        maximum_bytes: u64,
    },
    RetainSmallStages {
        maximum_bytes_per_stage: u64,
    },
    /// Retain whole existing chunks, largest first, within each stage ceiling.
    /// Unselected chunks retire through the ordinary pool maintenance path.
    RetainStageBudgets {
        maximum_bytes_by_stage: ImageStageRetentionBudgets,
    },
}
impl ImageDynamicResidencyPolicy {
    pub fn validate(self, residency: ImageResidencyPolicy) -> anyhow::Result<()> {
        residency.validate()?;
        if self == Self::Adaptive {
            ensure!(
                residency == ImageResidencyPolicy::Adaptive,
                "adaptive dynamic retention requires adaptive session residency"
            );
        }
        if let Self::RetainDecoder { maximum_bytes } = self {
            ensure!(
                maximum_bytes > 0,
                "decoder dynamic retention ceiling must be positive"
            );
            ensure!(
                residency.retains(ImageStage::Decoder),
                "decoder dynamic retention requires decoder session residency"
            );
        }
        if let Self::RetainSmallStages {
            maximum_bytes_per_stage,
        } = self
        {
            ensure!(
                maximum_bytes_per_stage > 0,
                "stage dynamic retention ceiling must be positive"
            );
            ensure!(
                residency != ImageResidencyPolicy::Cold,
                "stage dynamic retention requires resident sessions"
            );
        }
        if let Self::RetainStageBudgets {
            maximum_bytes_by_stage,
        } = self
        {
            let mut selected = false;
            for stage in RETAINABLE_STAGES {
                if maximum_bytes_by_stage.maximum_bytes(stage) > 0 {
                    selected = true;
                    ensure!(
                        residency.retains(stage),
                        "dynamic retention for {stage:?} requires its session residency"
                    );
                }
            }
            ensure!(
                selected,
                "stage retention budgets must select at least one stage"
            );
        }
        Ok(())
    }

    fn stage_ceiling(self, stage: ImageStage) -> u64 {
        match self {
            Self::Release => 0,
            Self::RetainDecoder { maximum_bytes } if stage == ImageStage::Decoder => maximum_bytes,
            Self::RetainSmallStages {
                maximum_bytes_per_stage,
            } => maximum_bytes_per_stage,
            Self::RetainStageBudgets {
                maximum_bytes_by_stage,
            } => maximum_bytes_by_stage.maximum_bytes(stage),
            _ => 0,
        }
    }
}

const RETAINABLE_STAGES: [ImageStage; 6] = [
    ImageStage::TextEncoder,
    ImageStage::VisionEncoder,
    ImageStage::VaeEncoder,
    ImageStage::Denoiser,
    ImageStage::Decoder,
    ImageStage::Noise,
];

fn resident_stage_index(stage: ImageStage) -> Option<usize> {
    RETAINABLE_STAGES.iter().position(|&kind| kind == stage)
}

fn same_resident_plan(left: &ImageStagePlanReceipt, right: &ImageStagePlanReceipt) -> bool {
    left.plan_hash == right.plan_hash
        && left.family_fingerprint == right.family_fingerprint
        && left.profile == right.profile
        && left.batch_size == right.batch_size
}

pub(super) struct ResidentStages<R: DeviceRuntime> {
    text_encoder: Option<ActiveImageStage<R>>,
    vision_encoder: Option<ActiveImageStage<R>>,
    vae_encoder: Option<ActiveImageStage<R>>,
    denoiser: Option<ActiveImageStage<R>>,
    decoder: Option<ActiveImageStage<R>>,
    noise: Option<ActiveImageStage<R>>,
    /// Last successful idle-retention receipts, including older shapes' pools.
    dynamic_bytes: [u64; 6],
    acquisition_order: [u64; 6],
    acquisition_clock: u64,
    text_encoder_weights: Option<ResidentWeights<R>>,
    vision_encoder_weights: Option<ResidentWeights<R>>,
    vae_encoder_weights: Option<ResidentWeights<R>>,
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
            vision_encoder: None,
            vae_encoder: None,
            denoiser: None,
            decoder: None,
            noise: None,
            dynamic_bytes: [0; 6],
            acquisition_order: [0; 6],
            acquisition_clock: 0,
            text_encoder_weights: None,
            vision_encoder_weights: None,
            vae_encoder_weights: None,
            denoiser_weights: None,
            decoder_weights: None,
        }
    }
    fn weights_slot(&mut self, stage: ImageStage) -> Option<&mut Option<ResidentWeights<R>>> {
        match stage {
            ImageStage::TextEncoder => Some(&mut self.text_encoder_weights),
            ImageStage::VisionEncoder => Some(&mut self.vision_encoder_weights),
            ImageStage::VaeEncoder => Some(&mut self.vae_encoder_weights),
            ImageStage::Denoiser => Some(&mut self.denoiser_weights),
            ImageStage::Decoder => Some(&mut self.decoder_weights),
            _ => None,
        }
    }
    fn slot(&mut self, stage: ImageStage) -> Option<&mut Option<ActiveImageStage<R>>> {
        match stage {
            ImageStage::TextEncoder => Some(&mut self.text_encoder),
            ImageStage::VisionEncoder => Some(&mut self.vision_encoder),
            ImageStage::VaeEncoder => Some(&mut self.vae_encoder),
            ImageStage::Denoiser => Some(&mut self.denoiser),
            ImageStage::Decoder => Some(&mut self.decoder),
            ImageStage::Noise => Some(&mut self.noise),
        }
    }
    fn dynamic_bytes(&self, stage: ImageStage) -> u64 {
        resident_stage_index(stage).map_or(0, |index| self.dynamic_bytes[index])
    }
    fn set_dynamic_bytes(&mut self, stage: ImageStage, bytes: u64) {
        if let Some(index) = resident_stage_index(stage) {
            self.dynamic_bytes[index] = bytes;
        }
    }
}

impl<R: DeviceRuntime> QwenImage21Pipeline<R> {
    pub fn residency_policy(&self) -> ImageResidencyPolicy {
        self.config.residency
    }

    fn runnable_dynamic_bytes(&self, prepared: &PreparedStage<R>) -> anyhow::Result<u64> {
        if self.config.residency.retains(prepared.plan.stage) {
            let previous = match prepared.plan.stage {
                ImageStage::TextEncoder => &self.resident_stages.text_encoder,
                ImageStage::VisionEncoder => &self.resident_stages.vision_encoder,
                ImageStage::VaeEncoder => &self.resident_stages.vae_encoder,
                ImageStage::Denoiser => &self.resident_stages.denoiser,
                ImageStage::Decoder => &self.resident_stages.decoder,
                ImageStage::Noise => &self.resident_stages.noise,
            };
            if let Some(previous) = previous {
                if same_resident_plan(&previous.receipt.plan, &prepared.plan) {
                    // Partial retention may keep an oversized pool while another
                    // pool is cold. Account for each pool before summing them.
                    return previous
                        .session
                        .minimum_runnable_dynamic_bytes_with_residency()
                        .context("resident stage dynamic footprint");
                }
            }
        }
        // A different plan closes its old session before allocating new pools.
        Ok(prepared
            .component
            .executable()
            .execution_plan()
            .payload()
            .memory()
            .minimum_runnable_request_bytes())
    }

    /// Batch planning evidence, not an allocation permit. All retained model
    /// installations coexist; only one stage's dynamic pools execute at once.
    /// Explicitly retained idle pools coexist with every other active stage.
    /// The shared device ledger remains authoritative under outside contention.
    #[cfg(test)]
    pub(super) fn validate_resident_capacity(
        &self,
        stages: &[&PreparedStage<R>],
    ) -> anyhow::Result<()> {
        self.validate_capacity_with_prefix(stages, None)
    }

    pub(super) fn validate_capacity_with_prefix(
        &self,
        stages: &[&PreparedStage<R>],
        prefix: Option<(&PreparedStage<R>, &PreparedStage<R>)>,
    ) -> anyhow::Result<()> {
        self.config
            .dynamic_residency
            .validate(self.config.residency)?;
        let adaptive = self.config.residency == ImageResidencyPolicy::Adaptive;
        let retained_dynamic_by_stage = RETAINABLE_STAGES.map(|stage| {
            if adaptive {
                return 0;
            }
            let ceiling = if self.config.residency.retains(stage) {
                self.config.dynamic_residency.stage_ceiling(stage)
            } else {
                0
            };
            ceiling.max(self.resident_stages.dynamic_bytes(stage))
        });
        let retained_dynamic = retained_dynamic_by_stage
            .iter()
            .try_fold(0_u64, |sum, bytes| {
                sum.checked_add(*bytes)
                    .context("retained dynamic allowances overflow")
            })?;
        let other_dynamic = |stage| {
            let own = if self.config.residency.retains(stage) {
                resident_stage_index(stage).map_or(0, |index| retained_dynamic_by_stage[index])
            } else {
                0
            };
            retained_dynamic - own
        };
        // Cold stages still coexist with the native allowance. Their static
        // bytes contribute only to the active peak, not a resident sum.
        let mut retained_static = 0_u64;
        let mut active_peak = 0_u64;
        let mut counted_static = Vec::new();
        for stage in stages {
            let payload = stage.component.executable().execution_plan().payload();
            ensure!(
                payload.work_limits() == RuntimeWorkLimits::FixedTensor,
                "image resident capacity requires compiled fixed tensor plans"
            );
            let memory = payload.memory();
            let retained = !adaptive && self.config.residency.retains(stage.plan.stage);
            if retained && !counted_static.contains(&stage.plan.stage) {
                retained_static = retained_static
                    .checked_add(memory.static_bytes())
                    .context("resident static bytes overflow")?;
                counted_static.push(stage.plan.stage);
            }
            // Fixed demand includes actual compiled provider scratch, per-pool
            // alignment, and proven activation/invocation lifetime reuse. Do
            // not add the cuDNN workspace limit again or sum all dynamic plans.
            // Prefix lifetimes are handled together below. In particular the
            // decode session is newly attached after the old slot is closed.
            let dynamic =
                if adaptive || (prefix.is_some() && stage.plan.stage == ImageStage::Denoiser) {
                    memory.minimum_runnable_request_bytes()
                } else {
                    self.runnable_dynamic_bytes(stage)?
                };
            let active = dynamic
                .checked_add(if retained { 0 } else { memory.static_bytes() })
                .and_then(|n| n.checked_add(other_dynamic(stage.plan.stage)))
                .context("retained pools plus active component bytes overflow")?;
            active_peak = active_peak.max(active);
        }
        if let Some((prefill, decode)) = prefix {
            let prefill_dynamic = if adaptive {
                prefill
                    .component
                    .executable()
                    .execution_plan()
                    .payload()
                    .memory()
                    .minimum_runnable_request_bytes()
            } else {
                self.runnable_dynamic_bytes(prefill)?
            };
            let prefill = prefill
                .component
                .executable()
                .execution_plan()
                .payload()
                .memory();
            let decode = decode
                .component
                .executable()
                .execution_plan()
                .payload()
                .memory();
            ensure!(
                prefill.static_bytes() >= decode.static_bytes(),
                "prefix decode must use a readonly subset of the prefill installation"
            );
            // Prefill output buffers and the compact decode Request state coexist
            // through the D2D fence. Counting only the larger plan misses one
            // whole cache. Decode Step backing is not admitted until prefill retires.
            let transition = prefill_dynamic
                .checked_add(decode.minimum_request_bytes())
                .and_then(|n| n.checked_add(decode.minimum_sequence_bytes()))
                .and_then(|n| n.checked_add(other_dynamic(ImageStage::Denoiser)))
                .and_then(|n| {
                    n.checked_add(
                        if !adaptive && self.config.residency.retains(ImageStage::Denoiser) {
                            0
                        } else {
                            prefill.static_bytes()
                        },
                    )
                })
                .context("prefix handoff capacity overflows")?;
            active_peak = active_peak.max(transition);
            // A subset attachment still retains and charges the original full
            // snapshot, including weights no longer consumed by decode.
            let decode_peak = decode
                .minimum_runnable_request_bytes()
                .checked_add(other_dynamic(ImageStage::Denoiser))
                .context("retained pools plus prefix decode capacity overflows")?
                .checked_add(
                    if !adaptive && self.config.residency.retains(ImageStage::Denoiser) {
                        0
                    } else {
                        prefill.static_bytes()
                    },
                )
                .context("prefix decode capacity overflows")?;
            active_peak = active_peak.max(decode_peak);
        }
        // A planning subset does not retire installations owned by other slots.
        // Count every live snapshot whose stage was not already accounted for.
        for (kind, weights) in [
            (
                ImageStage::TextEncoder,
                &self.resident_stages.text_encoder_weights,
            ),
            (
                ImageStage::VisionEncoder,
                &self.resident_stages.vision_encoder_weights,
            ),
            (
                ImageStage::VaeEncoder,
                &self.resident_stages.vae_encoder_weights,
            ),
            (ImageStage::Denoiser, &self.resident_stages.denoiser_weights),
            (ImageStage::Decoder, &self.resident_stages.decoder_weights),
        ] {
            if !adaptive && !counted_static.contains(&kind) {
                if let Some(weights) = weights {
                    retained_static = retained_static
                        .checked_add(weights.origin.plan.static_bytes)
                        .context("retained stage weights overflow")?;
                }
            }
        }
        let required = retained_static
            .checked_add(active_peak)
            .and_then(|n| n.checked_add(self.device_opaque_allowance_bytes))
            .context("resident batch capacity overflows")?;
        ensure!(required <= self.config.device_capacity_bytes,
            "resident image batch needs {required} device bytes (retained static {retained_static}, active stage peak {active_peak}, total idle dynamic allowance {retained_dynamic}, native allowance {}), exceeding total ceiling {}",
            self.device_opaque_allowance_bytes, self.config.device_capacity_bytes);
        Ok(())
    }

    /// The second phase binds this immutable installation while the first
    /// phase is still live. Cold residency uses a request-local shared owner.
    pub(super) fn start_prefix_prefill(
        &mut self,
        prepared: PreparedStage<R>,
        job: u64,
        stages: &mut Vec<ImageStageReceipt>,
    ) -> Result<(ActiveImageStage<R>, SharedComponentWeights<R>, bool), ImageGenerationFailure<R>>
    {
        if self.config.residency.retains(ImageStage::Denoiser) {
            let active = self.start_stage(prepared, job, stages)?;
            let weights = self
                .resident_stages
                .denoiser_weights
                .as_ref()
                .expect("resident denoiser installation")
                .weights
                .clone();
            Ok((active, weights, false))
        } else {
            let dit_source = ExactBf16TensorSource::new(self.assets.sources.transformer.as_ref());
            let native_source = ManagedDenseArchiveSource::new(
                self.assets.sources.transformer.as_ref(),
                &self.host,
            );
            let source =
                if QwenImage21TransformerFamily::is_fp8_profile(prepared.plan.profile.as_str()) {
                    ComponentWeightSource::Stored(&dit_source)
                } else {
                    ComponentWeightSource::Stored(&native_source)
                };
            prepared
                .start_shared(self, job, source)
                .map(|(active, weights)| (active, weights, true))
        }
    }

    pub(super) fn start_stage(
        &mut self,
        prepared: PreparedStage<R>,
        job: u64,
        stages: &mut Vec<ImageStageReceipt>,
    ) -> Result<ActiveImageStage<R>, ImageGenerationFailure<R>> {
        if self.config.residency == ImageResidencyPolicy::Adaptive {
            self.admit_adaptive_stage(&prepared, stages)?;
        }
        let weightless = matches!(prepared.component.family().weights(), ModelWeights::None);
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
                previous.receipt.weights_reused = !matches!(
                    previous
                        .session
                        .executable()
                        .execution_plan()
                        .payload()
                        .execution_weights(),
                    ExecutionWeights::None
                );
                if same_resident_plan(&previous.receipt.plan, &prepared.plan) {
                    previous.receipt.plan = prepared.plan;
                    return Ok(previous);
                }
                // Publish an actual zero-execution eviction receipt. A failed
                // close owns the original session and prevents initialization.
                // The old plan was compiled during an earlier acquisition.
                previous.receipt.plan.compile_wall_duration_us = 0;
                stages.push(previous.close()?);
                self.resident_stages
                    .set_dynamic_bytes(prepared.plan.stage, 0);
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
        if weightless {
            // A weightless contract has no Stored snapshot to share. Its normal
            // managed session may still retain idle dynamic pools between runs.
            return prepared.start(self, job, ComponentWeightSource::None);
        }
        // Keep the archive's optional implicit dtype conversion out of the
        // managed FP8 conversion path. These wrappers borrow the original mmap
        // and stay alive throughout synchronous static initialization.
        let text_source = ExactBf16TensorSource::new(self.assets.sources.text_encoder.as_ref());
        let dit_source = ExactBf16TensorSource::new(self.assets.sources.transformer.as_ref());
        let native_source =
            ManagedDenseArchiveSource::new(self.assets.sources.transformer.as_ref(), &self.host);
        let profile = prepared.plan.profile.as_str();
        let source = match prepared.plan.stage {
            ImageStage::TextEncoder if Qwen3VLTextEncoderFamily::is_fp8_profile(profile) => {
                ComponentWeightSource::Stored(&text_source)
            }
            ImageStage::TextEncoder => {
                ComponentWeightSource::Stored(self.assets.sources.text_encoder.as_ref())
            }
            ImageStage::Noise => ComponentWeightSource::None,
            ImageStage::VisionEncoder => {
                ComponentWeightSource::Stored(self.assets.sources.text_encoder.as_ref())
            }
            ImageStage::VaeEncoder => {
                ComponentWeightSource::Stored(self.assets.sources.vae.as_ref())
            }
            ImageStage::Denoiser if QwenImage21TransformerFamily::is_fp8_profile(profile) => {
                ComponentWeightSource::Stored(&dit_source)
            }
            ImageStage::Denoiser => ComponentWeightSource::Stored(&native_source),
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
                let started = std::time::Instant::now();
                let ceiling =
                    if self.config.dynamic_residency == ImageDynamicResidencyPolicy::Adaptive {
                        self.config.device_capacity_bytes
                    } else {
                        self.config
                            .dynamic_residency
                            .stage_ceiling(active.receipt.plan.stage)
                    };
                let retirement = if ceiling > 0 {
                    let outcome = match self.config.dynamic_residency {
                        ImageDynamicResidencyPolicy::RetainStageBudgets { .. }
                        | ImageDynamicResidencyPolicy::Adaptive => active
                            .session
                            .try_retain_largest_idle_dynamic_chunks(ceiling),
                        _ => active.session.try_retain_idle_dynamic_backing(ceiling),
                    };
                    match outcome {
                        Ok(IdleDynamicBackingOutcome::Retained(retention)) => {
                            ImageStageRetirement::ResidentBackingRetained { retention }
                        }
                        Ok(IdleDynamicBackingOutcome::Released(idle_release)) => {
                            ImageStageRetirement::Resident { idle_release }
                        }
                        Ok(IdleDynamicBackingOutcome::Busy(reason)) => {
                            return Err(active.processing(anyhow::anyhow!(
                                "stage is not idle for residency: {reason:?}"
                            )))
                        }
                        Err(error) => return Err(active.processing(error.into())),
                    }
                } else {
                    match active.session.try_release_idle_dynamic_backing() {
                        Ok(IdleDynamicReleaseOutcome::Released(idle_release)) => {
                            ImageStageRetirement::Resident { idle_release }
                        }
                        Ok(IdleDynamicReleaseOutcome::Busy(reason)) => {
                            return Err(active.processing(anyhow::anyhow!(
                                "stage is not idle for residency: {reason:?}"
                            )))
                        }
                        Err(error) => return Err(active.processing(error.into())),
                    }
                };
                let close_wall_duration_us = trace_elapsed_us(started);
                let stage = active.receipt.plan.stage;
                let retained_bytes = match &retirement {
                    ImageStageRetirement::ResidentBackingRetained { retention } => {
                        retention.retained_bytes()
                    }
                    _ => 0,
                };
                self.resident_stages
                    .set_dynamic_bytes(stage, retained_bytes);
                stages.push(ImageStageReceipt {
                    execution: active.receipt.clone(),
                    close: None,
                    retirement,
                    close_wall_duration_us,
                });
                *self
                    .resident_stages
                    .slot(stage)
                    .expect("resident stage slot") = Some(active);
                self.resident_stages.touch(stage);
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
    /// Converted-weight Host caches belong to the materializer/compiled-plan
    /// owners and remain charged until those owners, including this pipeline,
    /// are dropped; closing device sessions does not imply Host cache release.
    pub fn shutdown(&mut self) -> Result<Vec<ImageStageReceipt>, ImageGenerationError<R>> {
        self.stopped = true;
        let mut completed_stages = Vec::new();
        for stage in [
            ImageStage::TextEncoder,
            ImageStage::VisionEncoder,
            ImageStage::VaeEncoder,
            ImageStage::Denoiser,
            ImageStage::Decoder,
            ImageStage::Noise,
        ] {
            if let Some(mut active) = self.resident_stages.slot(stage).and_then(Option::take) {
                active.receipt.completed_executions = 0;
                active.receipt.execution_wall_durations_us.clear();
                active.receipt.initialization_wall_duration_us = 0;
                active.receipt.plan.compile_wall_duration_us = 0;
                active.receipt.session_reused = true;
                active.receipt.weights_reused = !matches!(
                    active
                        .session
                        .executable()
                        .execution_plan()
                        .payload()
                        .execution_weights(),
                    ExecutionWeights::None
                );
                match active.close() {
                    Ok(receipt) => {
                        completed_stages.push(receipt);
                        self.resident_stages.set_dynamic_bytes(stage, 0);
                    }
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
            ImageStage::VisionEncoder,
            ImageStage::VaeEncoder,
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
        // Successful root retirement precedes the actor's original lane owner.
        // External failure/completion owners keep their own lane authority;
        // releasing this owner alone never proves their quiescence.
        drop(self.serial_actor.take());
        Ok(completed_stages)
    }
}
