//! Bounded model session/snapshot slots. Compatible plans share the same sealed
//! physical weights; dynamic storage remains independently admitted.
use super::*;
use std::num::NonZeroU64;

mod adaptive;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ImageResidencyPolicy {
    #[default]
    Cold,
    /// Keep reusable sessions and static weights until a later compiled stage
    /// needs their capacity. Idle dynamic backing is reclaimed before weights.
    /// Prefer a sufficient set of idle weights with the lowest observed reload
    /// cost; stages without a reload observation use acquisition order.
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

    pub(super) fn retains(self, stage: ImageStage) -> bool {
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
    pub text_connector: bool,
    #[serde(skip_serializing_if = "is_false")]
    pub text_refiner: bool,
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
            ImageStage::TextConnector => self.text_connector,
            ImageStage::TextRefiner => self.text_refiner,
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
    pub text_connector: Option<NonZeroU64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_refiner: Option<NonZeroU64>,
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
            ImageStage::TextConnector => self.text_connector,
            ImageStage::TextRefiner => self.text_refiner,
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

const RETAINABLE_STAGES: [ImageStage; 8] = [
    ImageStage::TextEncoder,
    ImageStage::TextConnector,
    ImageStage::TextRefiner,
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
    text_connector: Option<ActiveImageStage<R>>,
    text_refiner: Option<ActiveImageStage<R>>,
    vision_encoder: Option<ActiveImageStage<R>>,
    vae_encoder: Option<ActiveImageStage<R>>,
    denoiser: Option<ActiveImageStage<R>>,
    decoder: Option<ActiveImageStage<R>>,
    noise: Option<ActiveImageStage<R>>,
    dynamic_bytes: [u64; 8],
    acquisition_order: [u64; 8],
    acquisition_clock: u64,
    bindings: BTreeMap<ImageStage, ImageWeightOwnerKey>,
    weights: BTreeMap<ImageWeightOwnerKey, Option<ResidentWeights<R>>>,
}
struct ResidentWeights<R: DeviceRuntime> {
    weights: SharedComponentWeights<R>,
    origin: ImageStageExecutionReceipt,
    pins: Arc<std::sync::atomic::AtomicUsize>,
}
impl<R: DeviceRuntime> ResidentStages<R> {
    pub(super) fn has_weights(&self, stage: ImageStage) -> bool {
        self.weights(stage).is_some()
    }
    fn owner_key(&self, stage: ImageStage) -> Option<ImageWeightOwnerKey> {
        self.bindings.get(&stage).copied()
    }
    fn has_pinned_weights(&self, stage: ImageStage) -> bool {
        self.weights(stage)
            .is_some_and(|w| w.pins.load(std::sync::atomic::Ordering::Acquire) != 0)
    }
    pub(super) fn new<S: ImagePipelineSourceAdapter>(sources: &S) -> Self {
        Self {
            text_encoder: None,
            text_connector: None,
            text_refiner: None,
            vision_encoder: None,
            vae_encoder: None,
            denoiser: None,
            decoder: None,
            noise: None,
            dynamic_bytes: [0; 8],
            acquisition_order: [0; 8],
            acquisition_clock: 0,
            bindings: RETAINABLE_STAGES
                .into_iter()
                .filter_map(|stage| sources.weight_owner(stage).map(|key| (stage, key)))
                .collect(),
            weights: BTreeMap::new(),
        }
    }
    fn weights_slot(&mut self, stage: ImageStage) -> Option<&mut Option<ResidentWeights<R>>> {
        let key = self.owner_key(stage)?;
        Some(self.weights.entry(key).or_default())
    }
    fn weights(&self, stage: ImageStage) -> Option<&ResidentWeights<R>> {
        self.weights
            .get(&self.owner_key(stage)?)
            .and_then(Option::as_ref)
    }
    fn slot(&mut self, stage: ImageStage) -> Option<&mut Option<ActiveImageStage<R>>> {
        Some(match stage {
            ImageStage::TextEncoder => &mut self.text_encoder,
            ImageStage::TextConnector => &mut self.text_connector,
            ImageStage::TextRefiner => &mut self.text_refiner,
            ImageStage::VisionEncoder => &mut self.vision_encoder,
            ImageStage::VaeEncoder => &mut self.vae_encoder,
            ImageStage::Denoiser => &mut self.denoiser,
            ImageStage::Decoder => &mut self.decoder,
            ImageStage::Noise => &mut self.noise,
        })
    }
    fn dynamic_bytes(&self, stage: ImageStage) -> u64 {
        resident_stage_index(stage).map_or(0, |i| self.dynamic_bytes[i])
    }
    fn set_dynamic_bytes(&mut self, stage: ImageStage, bytes: u64) {
        if let Some(i) = resident_stage_index(stage) {
            self.dynamic_bytes[i] = bytes;
        }
    }
}

impl<R: DeviceRuntime, S: ImagePipelineSourceAdapter> ImagePipelineExecution<R, S> {
    pub fn pin_weight_owner(
        &self,
        key: ImageWeightOwnerKey,
    ) -> anyhow::Result<ImageWeightOwnerPin<R>> {
        let weights = self
            .resident_stages
            .weights
            .get(&key)
            .and_then(Option::as_ref)
            .context("image weight owner is not installed")?;
        ImageWeightOwnerPin::new(key, weights.weights.clone(), weights.pins.clone())
    }
    pub fn residency_policy(&self) -> ImageResidencyPolicy {
        self.config.residency
    }

    fn runnable_dynamic_bytes(&self, prepared: &PreparedStage<R>) -> anyhow::Result<u64> {
        if self.config.residency.retains(prepared.plan.stage) {
            if let Some(previous) = self.resident_stages.stage(prepared.plan.stage) {
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
        self.capacity_with_prefix(stages, prefix)?.validate()?;
        Ok(())
    }

    pub(super) fn capacity_with_prefix(
        &self,
        stages: &[&PreparedStage<R>],
        prefix: Option<(&PreparedStage<R>, &PreparedStage<R>)>,
    ) -> anyhow::Result<ImageDeviceCapacityReport> {
        self.capacity_with_prefix_retention(stages, prefix, false)
    }

    /// Advisory steady-state demand if all weights required by this workload
    /// remain installed. Idle dynamic pools can still be retired. This is a
    /// scheduling preference, not the staged execution's admission boundary.
    pub(super) fn resident_preference_with_prefix(
        &self,
        stages: &[&PreparedStage<R>],
        prefix: Option<(&PreparedStage<R>, &PreparedStage<R>)>,
    ) -> anyhow::Result<ImageDeviceCapacityReport> {
        self.capacity_with_prefix_retention(stages, prefix, true)
    }

    fn capacity_with_prefix_retention(
        &self,
        stages: &[&PreparedStage<R>],
        prefix: Option<(&PreparedStage<R>, &PreparedStage<R>)>,
        retain_required_weights: bool,
    ) -> anyhow::Result<ImageDeviceCapacityReport> {
        self.config
            .dynamic_residency
            .validate(self.config.residency)?;
        let adaptive = self.config.residency == ImageResidencyPolicy::Adaptive;
        let retained_dynamic_by_stage = RETAINABLE_STAGES.map(|stage| {
            if !self.sources.supports_stage(stage) || adaptive {
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
        let mut counted_static =
            BTreeMap::<(Option<ImageWeightOwnerKey>, Option<ImageStage>), u64>::new();
        for stage in stages {
            stage.validate_declaration()?;
            let payload = stage.component.executable().execution_plan().payload();
            ensure!(
                payload.work_limits() == RuntimeWorkLimits::FixedTensor,
                "image resident capacity requires compiled fixed tensor plans"
            );
            let memory = payload.memory();
            let owner = stage.declaration().weight_owner;
            let key = (owner, owner.is_none().then_some(stage.plan.stage));
            let retained = retain_required_weights
                || (!adaptive && self.config.residency.retains(stage.plan.stage))
                || stage.declaration().attachment != ImageStageAttachment::MayInitializeStored
                || self.resident_stages.has_pinned_weights(stage.plan.stage);
            if retained {
                let bytes = self
                    .resident_stages
                    .weights(stage.plan.stage)
                    .map_or(memory.static_bytes(), |w| {
                        memory.static_bytes().max(w.origin.plan.static_bytes)
                    });
                let previous = counted_static.get(&key).copied().unwrap_or(0);
                let next = previous.max(bytes);
                retained_static = retained_static
                    .checked_add(next - previous)
                    .context("resident static bytes overflow")?;
                counted_static.insert(key, next);
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
                        if retain_required_weights
                            || (!adaptive && self.config.residency.retains(ImageStage::Denoiser))
                        {
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
                    if retain_required_weights
                        || (!adaptive && self.config.residency.retains(ImageStage::Denoiser))
                    {
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
        for (owner, weights) in &self.resident_stages.weights {
            let Some(weights) = weights else { continue };
            if !adaptive || weights.pins.load(std::sync::atomic::Ordering::Acquire) != 0 {
                let key = (Some(*owner), None);
                let previous = counted_static.get(&key).copied().unwrap_or(0);
                let next = previous.max(weights.origin.plan.static_bytes);
                retained_static = retained_static
                    .checked_add(next - previous)
                    .context("retained stage weights overflow")?;
                counted_static.insert(key, next);
            }
        }
        let required = retained_static
            .checked_add(active_peak)
            .and_then(|n| n.checked_add(self.device_opaque_allowance_bytes))
            .context("resident batch capacity overflows")?;
        Ok(ImageDeviceCapacityReport {
            complete_batch: true,
            required_bytes: required,
            capacity_bytes: if adaptive {
                self.adaptive_capacity_ceiling()?
            } else {
                self.config.device_capacity_bytes
            },
            retained_static_bytes: retained_static,
            active_stage_peak_bytes: active_peak,
            retained_dynamic_allowance_bytes: retained_dynamic,
            native_allowance_bytes: self.device_opaque_allowance_bytes,
        })
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
                .weights(ImageStage::Denoiser)
                .expect("resident denoiser installation")
                .weights
                .clone();
            Ok((active, weights, false))
        } else {
            let stored = self
                .sources
                .stage_source(
                    prepared.declaration(),
                    &prepared.plan.profile,
                    self.execution_host(),
                )
                .map_err(ImageGenerationFailure::Preflight)?;
            let source = match stored.as_ref() {
                Some(source) => ComponentWeightSource::Stored(source),
                None => ComponentWeightSource::None,
            };
            prepared
                .start_shared(self, job, source)
                .map(|(active, weights)| (active, weights, true))
        }
    }

    pub fn start_stage(
        &mut self,
        mut prepared: PreparedStage<R>,
        job: u64,
        stages: &mut Vec<ImageStageReceipt>,
    ) -> Result<ActiveImageStage<R>, ImageGenerationFailure<R>> {
        prepared
            .validate_declaration()
            .map_err(ImageGenerationFailure::Preflight)?;
        if self.is_stopped() {
            return Err(ImageGenerationFailure::Unavailable);
        }
        if prepared.declaration().attachment == ImageStageAttachment::RequireShared
            && !self.resident_stages.has_weights(prepared.plan.stage)
        {
            return Err(ImageGenerationFailure::Preflight(anyhow::anyhow!(
                "required shared image weight owner is absent"
            )));
        }
        if self.config.residency != ImageResidencyPolicy::Adaptive {
            return self.start_stage_once(prepared, job, stages);
        }
        loop {
            let retry = prepared.clone();
            match self.start_stage_once(prepared, job, stages) {
                Ok(active) => return Ok(active),
                Err(ImageGenerationFailure::Initialization {
                    plan,
                    source:
                        ComponentInitializationError::Pool {
                            source: VNextError::DeviceCapacityUnavailable(pressure),
                            transaction,
                            ..
                        },
                }) => {
                    // Native/process capacity can change after the advisory
                    // estimate (and some runtimes expose no free-memory query).
                    // Pool initialization has not uploaded or submitted work.
                    // Only its acknowledged journal release permits eviction
                    // and retry; failed cleanup retains the original journal.
                    match transaction.release() {
                        Ok(released) => drop(released),
                        Err(source) => {
                            return Err(ImageGenerationFailure::InitializationRelease {
                                plan,
                                source,
                            });
                        }
                    }
                    if !self.reclaim_after_allocation_failure(plan.stage, stages)? {
                        return Err(ImageGenerationFailure::InitializationCapacityRejected {
                            plan,
                            pressure,
                        });
                    }
                    tracing::warn!(stage=?plan.stage, job, %pressure,
                        "image pool initialization released; retrying after actual idle-owner reclamation");
                    prepared = retry;
                }
                Err(failure) => {
                    // Only a recoverable allocation transaction qualifies. No
                    // submitted inference, partial upload, or unknown fence is retried.
                    let (stage, failure) = failure.recover_allocation()?;
                    if !self.reclaim_after_allocation_failure(stage, stages)? {
                        return Err(ImageGenerationFailure::AllocationRejected { stage, failure });
                    }
                    tracing::warn!(?stage, job, error = %failure.message(),
                        "image allocation rolled back; retrying after idle residency reclamation");
                    prepared = retry;
                }
            }
        }
    }

    fn start_stage_once(
        &mut self,
        prepared: PreparedStage<R>,
        job: u64,
        stages: &mut Vec<ImageStageReceipt>,
    ) -> Result<ActiveImageStage<R>, ImageGenerationFailure<R>> {
        if self.config.residency == ImageResidencyPolicy::Adaptive {
            self.admit_adaptive_stage(&prepared, stages)?;
        }
        let declaration = *prepared.declaration();
        let stage = declaration.stage;
        let weightless = matches!(prepared.component.family().weights(), ModelWeights::None);
        if !weightless && declaration.weight_owner.is_none() {
            return Err(ImageGenerationFailure::Preflight(anyhow::anyhow!(
                "weighted stage has no declared weight owner"
            )));
        }
        if self.config.residency.retains(stage) {
            if let Some(mut previous) = self.resident_stages.slot(stage).and_then(Option::take) {
                previous.receipt.completed_executions = 0;
                previous.receipt.execution_wall_durations_us.clear();
                previous.receipt.device_profiles.clear();
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
                    previous.declaration = declaration;
                    // Retirement detaches the preceding request's Output
                    // view. Reacquiring its real executor must consume this
                    // request's admitted grant, just as a fresh attachment
                    // does, rather than allocating readbacks on the parent.
                    if let Err(source) = previous
                        .session
                        .set_output_host_account(self.execution_host().clone())
                    {
                        return Err(previous.processing(source.into()));
                    }
                    return Ok(previous);
                }
                previous.receipt.plan.compile_wall_duration_us = 0;
                stages.push(previous.close()?);
                self.resident_stages.set_dynamic_bytes(stage, 0);
            }
        }
        // This is the exact sealed physical owner, including a First graph's
        // larger installation when a refiner/steady subset attaches to it.
        if let Some(weights) = self
            .resident_stages
            .weights(stage)
            .map(|w| w.weights.clone())
        {
            return prepared.start_attached(self, job, &weights);
        }
        if declaration.attachment == ImageStageAttachment::RequireShared {
            return Err(ImageGenerationFailure::Preflight(anyhow::anyhow!(
                "required shared image weight owner is absent"
            )));
        }
        if weightless {
            return prepared.start(self, job, ComponentWeightSource::None);
        }
        let stored = self
            .sources
            .stage_source(&declaration, &prepared.plan.profile, self.execution_host())
            .map_err(ImageGenerationFailure::Preflight)?;
        let source = match stored.as_ref() {
            Some(source) => ComponentWeightSource::Stored(source),
            None => ComponentWeightSource::None,
        };
        if self.config.residency.retains(stage)
            || declaration.attachment == ImageStageAttachment::InitializeShared
        {
            let (active, weights) = prepared.start_shared(self, job, source)?;
            drop(stored);
            *self
                .resident_stages
                .weights_slot(stage)
                .expect("declared weighted stage") = Some(ResidentWeights {
                weights,
                origin: active.receipt.clone(),
                pins: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            });
            Ok(active)
        } else {
            prepared.start(self, job, source)
        }
    }

    pub fn finish_stage(
        &mut self,
        mut active: ActiveImageStage<R>,
        stages: &mut Vec<ImageStageReceipt>,
    ) -> Result<(), ImageGenerationFailure<R>> {
        if active.receipt.plan.stage != active.declaration.stage
            || active.receipt.plan.role != active.declaration.role
            || active.receipt.plan.batch_size != active.declaration.batch_size
        {
            return Err(active.processing(anyhow::anyhow!(
                "active image receipt differs from its sealed stage declaration"
            )));
        }
        // Resident executors must not carry a closed request's scoped view
        // into the next batch. The root still proves the same Host provenance.
        if let Err(error) = active.session.set_output_host_account(self.host.clone()) {
            return Err(active.processing(error.into()));
        }
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
        for stage in RETAINABLE_STAGES {
            if let Some(mut active) = self.resident_stages.slot(stage).and_then(Option::take) {
                active.receipt.completed_executions = 0;
                active.receipt.execution_wall_durations_us.clear();
                active.receipt.device_profiles.clear();
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
        for stage in RETAINABLE_STAGES {
            if let Some(ResidentWeights {
                weights,
                mut origin,
                ..
            }) = self
                .resident_stages
                .weights_slot(stage)
                .and_then(Option::take)
            {
                origin.completed_executions = 0;
                origin.execution_wall_durations_us.clear();
                origin.device_profiles.clear();
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
        if let Some(trace) = &self.continuous_trace {
            // End of this producer, not a claim about global GPU memory or
            // separately retained failure owners.
            trace.seal();
        }
        Ok(completed_stages)
    }
}
