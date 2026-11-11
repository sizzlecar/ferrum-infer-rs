//! One execution owner for image components. Model adapters select exact
//! Stored sources; runtime, Host grants, plans and resident native owners are
//! shared by all of that pipeline's stages.
use super::*;
use ferrum_quantization::{
    bf16_whole_tensor_fp8::ExactBf16TensorSource, ManagedDenseArchiveSource,
};
mod declaration;
mod pin;
mod qwen_bridge;
pub use declaration::{ImageStageAttachment, ImageStageDeclaration, ImageWeightOwnerKey};
pub use pin::ImageWeightOwnerPin;

/// A fixed pipeline's immutable weight-source domain. Selecting a source does
/// not materialize an entire model, create another runtime, or grant capacity.
pub trait ImagePipelineSourceAdapter: Clone + Send + Sync + 'static {
    type Source<'a>: WeightComponentSource + 'a
    where
        Self: 'a;
    fn stage_source<'a>(
        &'a self,
        declaration: &ImageStageDeclaration,
        profile: &NumericalProfileId,
        host: &'a HostMemoryAccount,
    ) -> anyhow::Result<Option<Self::Source<'a>>>;
    fn weight_owner(&self, stage: ImageStage) -> Option<ImageWeightOwnerKey>;
    fn supports_stage(&self, stage: ImageStage) -> bool {
        stage == ImageStage::Noise || self.weight_owner(stage).is_some()
    }
    /// Additional physical source/materializer backing, retaining any warm
    /// assumptions through installation. Unknown conversions must be rejected.
    fn prepare_host_backing(
        &self,
        declaration: &ImageStageDeclaration,
        profile: &NumericalProfileId,
        family: &PreparedModelFamily,
        plan: &ExecutionPlan,
        host: &HostMemoryAccount,
    ) -> anyhow::Result<(usize, HostBackingPreparation)>;
}

pub struct ImagePipelineExecution<R: DeviceRuntime, S: ImagePipelineSourceAdapter> {
    pub(super) runtime: Arc<R>,
    pub(super) host: HostMemoryAccount,
    pub(super) execution_host: Option<HostMemoryAccount>,
    pub(super) registry: Arc<OperationRuntimeRegistry<R>>,
    pub(super) catalog: CapabilityCatalog,
    pub(super) materializers: Arc<WeightMaterializerRegistry>,
    pub(super) config: ImagePipelineConfig,
    pub(super) policy: TensorComponentPolicy,
    pub(super) next_job: u64,
    pub(super) continuous_trace: Option<super::continuous_trace::ImageContinuousTraceRecorder>,
    pub(super) stopped: bool,
    pub(super) service_preparation_identity: Arc<()>,
    pub(super) plan_cache: Arc<plan_cache::StagePlanCache<R>>,
    pub(super) preparation_only: bool,
    pub(super) resident_stages: residency::ResidentStages<R>,
    pub(super) device_opaque_allowance_bytes: u64,
    pub(super) sources: S,
    // Release resident sessions before their one original serial lane owner.
    pub(super) serial_actor: Option<TensorSerialActor<R>>,
}
impl<R: DeviceRuntime, S: ImagePipelineSourceAdapter> ImagePipelineExecution<R, S> {
    pub(super) fn retained_diagnostic_output_bytes(&self) -> usize {
        self.continuous_trace
            .as_ref()
            .map_or(0, |trace| trace.backing_bytes())
    }

    /// A request must fit beside the producer's fixed diagnostic backing.
    /// Only this hard-ceiling proof includes that existing allocation; actual
    /// scope admission/reservation must continue to request only new work.
    pub(super) fn check_scope_capacity_with_retained_diagnostics(
        &self,
        claims: &[(HostMemoryClass, usize)],
    ) -> Result<(), ferrum_runtime::HostMemoryError> {
        let retained = self.retained_diagnostic_output_bytes();
        if retained == 0 {
            return self.host.check_scope_capacity(claims);
        }
        let mut whole = Vec::with_capacity(claims.len() + 1);
        whole.extend_from_slice(claims);
        whole.push((HostMemoryClass::Output, retained));
        self.host.check_scope_capacity(&whole)
    }

    pub fn compile_stage(
        &self,
        declaration: ImageStageDeclaration,
        family: PreparedModelFamily,
        options: &ProgramPlanCompileOptions,
    ) -> anyhow::Result<PreparedImageStage<R>> {
        PreparedStage::compile_declared(declaration, family, options, self)
    }
    pub fn stage_host_backing(
        &self,
        stage: &PreparedImageStage<R>,
    ) -> anyhow::Result<(usize, HostBackingPreparation)> {
        stage.validate_declaration()?;
        if stage.declaration().attachment == ImageStageAttachment::RequireShared {
            // The declared shared-only path forbids source reads. Starting still
            // requires the actual snapshot, even when metadata is prepared
            // before its First installation.
            return Ok((0, HostBackingPreparation::new(0)));
        }
        // Native residency may change after this metadata calculation. A
        // normal cold-capable stage must retain its source's conservative
        // bound; only the source/materializer's retained warm Host preparation
        // can safely reduce additional conversion/cache credit.
        self.sources.prepare_host_backing(
            stage.declaration(),
            &stage.plan.profile,
            stage.component.family(),
            stage.component.executable().execution_plan(),
            &self.host,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        runtime: Arc<R>,
        host: HostMemoryAccount,
        registry: OperationRuntimeRegistry<R>,
        catalog: CapabilityCatalog,
        materializers: WeightMaterializerRegistry,
        config: ImagePipelineConfig,
        sources: S,
    ) -> anyhow::Result<Self> {
        config.validate()?;
        if let (Some(text), Some(connector)) = (
            sources.weight_owner(ImageStage::TextEncoder),
            sources.weight_owner(ImageStage::TextConnector),
        ) {
            ensure!(
                text != connector,
                "text encoder and connector require distinct weight-owner keys"
            );
        }
        ensure!(
            config.device_capacity_bytes <= runtime.descriptor().total_memory_bytes,
            "component capacity exceeds physical device descriptor"
        );
        ensure!(
            catalog.device() == runtime.descriptor(),
            "catalog and actual runtime device descriptors differ"
        );
        let policy = TensorComponentPolicy::new(
            config.device_capacity_bytes,
            runtime
                .descriptor()
                .dynamic_storage_profiles
                .iter()
                .copied()
                .collect(),
        )?;
        let serial_actor = TensorSerialActor::new(runtime.clone())?;
        Ok(Self {
            runtime,
            host,
            execution_host: None,
            registry: Arc::new(registry),
            catalog,
            materializers: Arc::new(materializers),
            config,
            policy,
            next_job: 0,
            continuous_trace: None,
            stopped: false,
            service_preparation_identity: Arc::new(()),
            plan_cache: Arc::new(plan_cache::StagePlanCache::new()),
            preparation_only: false,
            resident_stages: residency::ResidentStages::new(&sources),
            device_opaque_allowance_bytes: 0,
            sources,
            serial_actor: Some(serial_actor),
        })
    }
    pub fn runtime(&self) -> &Arc<R> {
        &self.runtime
    }
    pub fn is_stopped(&self) -> bool {
        self.stopped
            || self
                .serial_actor
                .as_ref()
                .is_none_or(TensorSerialActor::is_failed)
    }
    pub fn host_account(&self) -> &HostMemoryAccount {
        &self.host
    }
    pub(super) fn execution_host(&self) -> &HostMemoryAccount {
        self.execution_host.as_ref().unwrap_or(&self.host)
    }
    pub(super) fn preparation_fork(&self, capacity: u64) -> Self {
        Self {
            runtime: self.runtime.clone(),
            host: self.host.clone(),
            execution_host: self.execution_host.clone(),
            registry: self.registry.clone(),
            catalog: self.catalog.clone(),
            materializers: self.materializers.clone(),
            config: ImagePipelineConfig {
                device_capacity_bytes: capacity,
                ..self.config.clone()
            },
            policy: self.policy.clone(),
            next_job: self.next_job.saturating_add(1),
            // Preparation forks neither execute nor publish actual cohorts.
            continuous_trace: None,
            stopped: false,
            service_preparation_identity: self.service_preparation_identity.clone(),
            plan_cache: self.plan_cache.clone(),
            preparation_only: true,
            resident_stages: residency::ResidentStages::new(&self.sources),
            device_opaque_allowance_bytes: self.device_opaque_allowance_bytes,
            sources: self.sources.clone(),
            serial_actor: None,
        }
    }
}

#[derive(Clone)]
pub(super) struct QwenImage21SourceAdapter {
    pub(super) assets: Arc<QwenImage21Assets>,
}
pub(super) enum QwenImage21StageSource<'a> {
    Archive(&'a dyn TensorArchive),
    Exact(ExactBf16TensorSource<'a>),
    Managed(ManagedDenseArchiveSource<'a>),
}
impl WeightComponentSource for QwenImage21StageSource<'_> {
    fn component<'a>(
        &'a self,
        spec: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        match self {
            Self::Archive(v) => v.component(spec),
            Self::Exact(v) => v.component(spec),
            Self::Managed(v) => v.component(spec),
        }
    }
    fn component_segments<'a>(
        &'a self,
        spec: &WeightComponentSpec,
    ) -> Result<WeightComponentSegments<'a>, VNextError> {
        match self {
            Self::Archive(v) => v.component_segments(spec),
            Self::Exact(v) => v.component_segments(spec),
            Self::Managed(v) => v.component_segments(spec),
        }
    }
}
impl ImagePipelineSourceAdapter for QwenImage21SourceAdapter {
    type Source<'a> = QwenImage21StageSource<'a>;
    fn stage_source<'a>(
        &'a self,
        declaration: &ImageStageDeclaration,
        profile: &NumericalProfileId,
        host: &'a HostMemoryAccount,
    ) -> anyhow::Result<Option<Self::Source<'a>>> {
        let sources = &self.assets.sources;
        Ok(Some(match declaration.stage {
            ImageStage::TextConnector | ImageStage::TextRefiner => {
                anyhow::bail!("Qwen image source does not declare this stage")
            }
            ImageStage::Noise => return Ok(None),
            ImageStage::TextEncoder
                if Qwen3VLTextEncoderFamily::is_fp8_profile(profile.as_str()) =>
            {
                QwenImage21StageSource::Exact(ExactBf16TensorSource::new(
                    sources.text_encoder.as_ref(),
                ))
            }
            ImageStage::TextEncoder | ImageStage::VisionEncoder => {
                QwenImage21StageSource::Archive(sources.text_encoder.as_ref())
            }
            ImageStage::VaeEncoder | ImageStage::Decoder => {
                QwenImage21StageSource::Archive(sources.vae.as_ref())
            }
            ImageStage::Denoiser
                if QwenImage21TransformerFamily::is_fp8_profile(profile.as_str()) =>
            {
                QwenImage21StageSource::Exact(ExactBf16TensorSource::new(
                    sources.transformer.as_ref(),
                ))
            }
            ImageStage::Denoiser => QwenImage21StageSource::Managed(
                ManagedDenseArchiveSource::new(sources.transformer.as_ref(), host),
            ),
        }))
    }
    fn weight_owner(&self, stage: ImageStage) -> Option<ImageWeightOwnerKey> {
        Some(match stage {
            ImageStage::TextEncoder => ImageWeightOwnerKey::TextEncoder,
            ImageStage::VisionEncoder => ImageWeightOwnerKey::VisionEncoder,
            ImageStage::VaeEncoder => ImageWeightOwnerKey::VaeEncoder,
            ImageStage::Denoiser => ImageWeightOwnerKey::Denoiser,
            ImageStage::Decoder => ImageWeightOwnerKey::Decoder,
            ImageStage::TextConnector | ImageStage::TextRefiner | ImageStage::Noise => return None,
        })
    }
    fn prepare_host_backing(
        &self,
        declaration: &ImageStageDeclaration,
        profile: &NumericalProfileId,
        family: &PreparedModelFamily,
        plan: &ExecutionPlan,
        host: &HostMemoryAccount,
    ) -> anyhow::Result<(usize, HostBackingPreparation)> {
        let source = match declaration.stage {
            ImageStage::TextEncoder | ImageStage::VisionEncoder => {
                self.assets.sources.text_encoder.as_ref()
            }
            ImageStage::Denoiser => self.assets.sources.transformer.as_ref(),
            ImageStage::VaeEncoder | ImageStage::Decoder => self.assets.sources.vae.as_ref(),
            ImageStage::Noise => {
                ensure!(
                    matches!(family.weights(), ModelWeights::None),
                    "noise source requires weightless family"
                );
                let backing = plan.prepare_host_backing(family, None)?;
                return Ok((backing.additional_bytes(), backing));
            }
            ImageStage::TextConnector | ImageStage::TextRefiner => {
                anyhow::bail!("Qwen image source does not declare this stage")
            }
        };
        let exact = ExactBf16TensorSource::new(source);
        let backing = plan.prepare_host_backing(family, Some(&exact))?;
        let converted = backing.additional_bytes();
        if declaration.stage != ImageStage::Denoiser
            || QwenImage21TransformerFamily::is_fp8_profile(profile.as_str())
        {
            return Ok((converted, backing));
        }
        // This legacy Qwen dense source materializes one identity component at
        // a time. Other model sources provide their own group-aware proof.
        let mut normalized = 0;
        for component in &family.stored_weight_schema()?.components {
            let [name] = component.external_names.as_slice() else {
                continue;
            };
            if source.tensor(name)?.encoding() != &component.encoding {
                normalized = normalized
                    .max(host.backing_bytes(usize::try_from(component.physical_bytes()?)?)?);
            }
        }
        Ok((
            converted
                .checked_add(normalized)
                .context("source Host backing extent overflows")?,
            backing,
        ))
    }
}

#[cfg(test)]
mod tests;
