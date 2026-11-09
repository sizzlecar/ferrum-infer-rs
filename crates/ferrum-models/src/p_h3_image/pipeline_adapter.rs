//! Public component preparation for an injected image execution engine. This
//! facade owns immutable model/source metadata, never a runtime, actor, device
//! reservation or Host ledger. Internal DiT/stack/block schemas stay private.
pub use super::dit_forward::PH3ImageDenoiserAttentionPolicy;
use super::dit_forward::{
    self, PH3ImageDiTConfig, PH3ImageDiTFamily, PH3ImageDiTLimits, PH3ImageDiTPhase,
    PH3ImageDiTRequest,
};
pub use super::dit_forward::{
    PH3ImageDiTConditioningKey as PH3ImageConditioningKey, PH3ImageRefinedTextCache,
};
pub use super::dit_forward::{PH3ImageRefinementDestination, PH3ImageTextRefinementError};
use super::*;
use anyhow::{ensure, Context, Result};
use ferrum_execution::{
    component::{ComponentInitializationReceipt, TensorComponentSession},
    TensorSerialActor,
};
use ferrum_quantization::comfy_convrot::ComfyCanonicalWeightSource;
use ferrum_runtime::*;
use std::{collections::BTreeSet, sync::Arc};
mod family_preparation;
mod refinement_host_requirements;
pub use family_preparation::{PH3ImageDenoiserFamilyConfig, PH3ImageDenoiserFamilyProvider};
pub use refinement_host_requirements::{
    PH3ImageRefinementCacheHostRequirement, PH3ImageRefinementHostRequirements,
};

#[derive(Clone, Copy, Debug)]
pub struct PH3ImageDenoiserLimits {
    pub max_metadata_bytes: usize,
    pub max_plan_nodes: usize,
    pub max_stored_bytes: u64,
}
impl From<PH3ImageDenoiserLimits> for PH3ImageDiTLimits {
    fn from(l: PH3ImageDenoiserLimits) -> Self {
        Self {
            max_metadata_bytes: l.max_metadata_bytes,
            max_plan_nodes: l.max_plan_nodes,
            max_stored_bytes: l.max_stored_bytes,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PH3ImageDenoiserPhase {
    InstallFullWeights,
    RefineFreshText,
    DenoiseRefinedText,
}
impl PH3ImageDenoiserPhase {
    fn internal(self) -> PH3ImageDiTPhase {
        match self {
            Self::InstallFullWeights => PH3ImageDiTPhase::WithTextRefiner,
            Self::RefineFreshText => PH3ImageDiTPhase::RefineOnly,
            Self::DenoiseRefinedText => PH3ImageDiTPhase::RefinedText,
        }
    }
}
/// Request identity and target geometry survive preparation; callers cannot
/// edit private modality/time RLE independently after it has been checked.
pub struct PH3ImageDenoiserRequest {
    identity: RequestIdentity,
    target: PH3ImageReferenceImage,
    inner: PH3ImageDiTRequest,
}
impl PH3ImageDenoiserRequest {
    pub fn identity(&self) -> &RequestIdentity {
        &self.identity
    }
}

pub struct PH3ImageDenoiserAdapter {
    transformer: PH3ImageTransformerConfig,
    source: Arc<ComfyCanonicalWeightSource>,
    curve: PH3ImageCurveTimeEmbedding,
    limits: PH3ImageDiTLimits,
    schema_fingerprint: String,
    stored_bytes: u64,
    attention_policy: PH3ImageDenoiserAttentionPolicy,
}
impl PH3ImageDenoiserAdapter {
    /// Validate original physical weights once. The smallest complete request
    /// is only a metadata recipe for schema identity, never a runtime fixture
    /// or a claim that the released model fits/executed on any device.
    pub fn new(
        transformer: PH3ImageTransformerConfig,
        source: Arc<ComfyCanonicalWeightSource>,
        limits: PH3ImageDenoiserLimits,
    ) -> Result<Self> {
        let limits: PH3ImageDiTLimits = limits.into();
        PH3ImageCheckpointRotary::new(&transformer, &source)?;
        let curve_tensor = source.archive().tensor("adaln_t_table")?;
        let [grid, basis] = curve_tensor.shape() else {
            anyhow::bail!("DiT curve source requires rank2 original table")
        };
        ensure!(
            curve_tensor.element_type() == Some(ElementType::F32),
            "DiT curve source must remain original F32"
        );
        let curve =
            PH3ImageCurveTimeEmbedding::new(usize::try_from(*grid)?, usize::try_from(*basis)?)?;
        let [_, h, w] = transformer.patch_size();
        let target = PH3ImageReferenceImage::new(usize::try_from(h)?, usize::try_from(w)?)?;
        let tags = [PH3ImageModality::Text];
        let layout = PH3ImageRequestLayout::new(&transformer, &tags, target, &[])?;
        let times = vec![0; layout.rows()];
        let request = PH3ImageDiTRequest::new(
            &transformer,
            &tags,
            target,
            None,
            &times,
            limits.max_metadata_bytes,
        )?;
        let config = PH3ImageDiTConfig::from_checkpoint(
            &transformer,
            curve,
            1,
            vec![request],
            PH3ImageDiTPhase::WithTextRefiner,
            &source,
            limits,
        )?;
        dit_forward::PH3ImageDiTCheckpoint::new(&config, &source, limits)?;
        let family = PH3ImageDiTFamily::new(limits)?;
        let ModelWeights::Stored(schema) = family.weights(&config)? else {
            anyhow::bail!("DiT adapter requires stored physical weights")
        };
        let schema_fingerprint = family_preparation::physical_inventory_fingerprint(&schema)?;
        let stored_bytes = schema.components.iter().try_fold(0u64, |sum, c| {
            sum.checked_add(c.physical_bytes()?)
                .context("DiT stored component byte total overflow")
        })?;
        Ok(Self {
            transformer,
            source,
            curve,
            limits,
            schema_fingerprint,
            stored_bytes,
            attention_policy: PH3ImageDenoiserAttentionPolicy::Sequential,
        })
    }
    /// Select arithmetic before constructing the pipeline. Preparation and
    /// refinement keys retain this explicit policy; weight schema is unchanged.
    pub fn set_attention_policy(&mut self, policy: PH3ImageDenoiserAttentionPolicy) -> Result<()> {
        ensure!(
            policy == PH3ImageDenoiserAttentionPolicy::Sequential
                || matches!(self.transformer.attention_head_dim(), 72 | 128),
            "cuDNN segmented SDPA requires head dimension 72 or 128"
        );
        self.attention_policy = policy;
        Ok(())
    }
    /// Actual Stored component payload bytes. Adapter table/frequency metadata,
    /// archive headers, device alignment and workspaces are separate owners.
    pub fn stored_bytes(&self) -> u64 {
        self.stored_bytes
    }
    pub fn transformer(&self) -> &PH3ImageTransformerConfig {
        &self.transformer
    }
    pub fn curve(&self) -> PH3ImageCurveTimeEmbedding {
        self.curve
    }
    pub fn request(
        &self,
        identity: RequestIdentity,
        tags: &[PH3ImageModality],
        target: PH3ImageReferenceImage,
        reference: Option<PH3ImageReferenceImage>,
        time_indices: &[u64],
    ) -> Result<PH3ImageDenoiserRequest> {
        Ok(PH3ImageDenoiserRequest {
            identity,
            target,
            inner: PH3ImageDiTRequest::new(
                &self.transformer,
                tags,
                target,
                reference,
                time_indices,
                self.limits.max_metadata_bytes,
            )?,
        })
    }
    pub fn prepare(
        &self,
        phase: PH3ImageDenoiserPhase,
        basis_rows: u64,
        requests: Vec<PH3ImageDenoiserRequest>,
    ) -> Result<PreparedPH3ImageDenoiser> {
        self.prepare_with_family(phase, basis_rows, requests, |provider, config, profile| {
            Ok(TypedFamilyRegistration::new(provider)
                .prepare_with_profile(&serde_json::to_value(config)?, profile)?)
        })
    }

    /// Use the caller's original family cache. Request identities remain in the
    /// request-local result, outside the immutable provider configuration.
    /// Schema, profile, geometry and metadata bounds are checked on every hit.
    pub fn prepare_with_family(
        &self,
        phase: PH3ImageDenoiserPhase,
        basis_rows: u64,
        requests: Vec<PH3ImageDenoiserRequest>,
        prepare_family: impl FnOnce(
            PH3ImageDenoiserFamilyProvider,
            &PH3ImageDenoiserFamilyConfig,
            &NumericalProfileId,
        ) -> Result<PreparedModelFamily>,
    ) -> Result<PreparedPH3ImageDenoiser> {
        ensure!(
            !requests.is_empty()
                && requests
                    .iter()
                    .map(|r| &r.identity)
                    .collect::<BTreeSet<_>>()
                    .len()
                    == requests.len(),
            "DiT stage requires distinct nonempty request identities"
        );
        let metadata = requests.iter().try_fold(0usize, |sum, r| {
            sum.checked_add(std::mem::size_of::<PH3ImageDenoiserRequest>())
                .and_then(|n| n.checked_add(r.identity.as_str().len()))
                .context("DiT request identity metadata overflow")
        })?;
        ensure!(
            metadata <= self.limits.max_metadata_bytes,
            "DiT request identity metadata exceeds capacity"
        );
        let identities = requests
            .iter()
            .map(|r| r.identity.clone())
            .collect::<Vec<_>>();
        let velocity = requests
            .iter()
            .map(|r| PH3ImageVelocityRequest::new(r.identity.clone(), r.target))
            .collect::<Vec<_>>();
        let config = PH3ImageDiTConfig::from_checkpoint(
            &self.transformer,
            self.curve,
            basis_rows,
            requests.into_iter().map(|r| r.inner).collect(),
            phase.internal(),
            &self.source,
            self.limits,
        )?
        .with_attention_policy(self.attention_policy)?;
        let provider = PH3ImageDiTFamily::new(self.limits)?;
        let options = provider.compile_options(&config)?;
        let profile = NumericalProfileId::new(PH3ImageDiTFamily::cuda_profile_id(&config))
            .map_err(anyhow::Error::msg)?;
        let provider = PH3ImageDenoiserFamilyProvider(provider);
        let family_id = provider.family_id().clone();
        let metadata_ids = provider.external_metadata_ids();
        let config = PH3ImageDenoiserFamilyConfig(config);
        let family = prepare_family(provider, &config, &profile)?;
        ensure!(
            family.family_id() == &family_id
                && metadata_ids.contains(family.external_metadata_id())
                && family.canonical_config() == &serde_json::to_value(&config)?
                && family.numerical_profile().id == profile,
            "cached DiT family differs from its validated geometry or numerical/source identity"
        );
        let schema = family.stored_weight_schema()?;
        ensure!(
            family_preparation::physical_inventory_fingerprint(schema)? == self.schema_fingerprint,
            "activation preparation changed the validated physical weight schema"
        );
        ensure!(
            serde_json::to_vec(&(&family, options.tensor_specs()))?
                .len()
                .checked_add(metadata)
                .is_some_and(|n| n <= self.limits.max_metadata_bytes),
            "prepared DiT family/identity metadata exceeds capacity"
        );
        Ok(PreparedPH3ImageDenoiser {
            config: config.0,
            family,
            options,
            phase,
            identities,
            velocity,
            source: self.source.clone(),
            transformer: self.transformer.clone(),
            curve: self.curve,
            limits: self.limits,
        })
    }
}

pub struct PreparedPH3ImageDenoiser {
    config: PH3ImageDiTConfig,
    family: PreparedModelFamily,
    options: ProgramPlanCompileOptions,
    phase: PH3ImageDenoiserPhase,
    identities: Vec<RequestIdentity>,
    velocity: Vec<PH3ImageVelocityRequest>,
    source: Arc<ComfyCanonicalWeightSource>,
    transformer: PH3ImageTransformerConfig,
    curve: PH3ImageCurveTimeEmbedding,
    limits: PH3ImageDiTLimits,
}
impl PreparedPH3ImageDenoiser {
    pub fn family(&self) -> &PreparedModelFamily {
        &self.family
    }
    pub fn compile_options(&self) -> &ProgramPlanCompileOptions {
        &self.options
    }
    pub fn source(&self) -> &ComfyCanonicalWeightSource {
        &self.source
    }
    pub fn phase(&self) -> PH3ImageDenoiserPhase {
        self.phase
    }
    pub fn request_ids(&self) -> &[RequestIdentity] {
        &self.identities
    }
    /// Original checked routing records, borrowed in the same order as
    /// `request_ids`. Observation does not rebuild or modify the graph.
    pub fn request_row_runs(
        &self,
    ) -> impl ExactSizeIterator<Item = &[PH3ImageCurveBlockRowRun]> + '_ {
        self.config.request_row_runs()
    }
    /// Initialization plans install all weights. Fresh text/steady plans must
    /// explicitly attach that sealed handle; no cold fallback occurs here.
    pub fn validate_attached_session<R: DeviceRuntime>(
        &self,
        session: &TensorComponentSession<R>,
        weights: &SharedComponentWeights<R>,
    ) -> Result<()> {
        let plan = session.executable().execution_plan().payload();
        ensure!(
            plan.family_id() == self.family.family_id()
                && plan.prepared_family_fingerprint() == self.family.fingerprint()?
                && plan.program_fingerprint() == self.family.program().fingerprint()?,
            "attached session differs from the prepared DiT stage"
        );
        let ComponentInitializationReceipt::SharedWeights { evidence } =
            session.initialization_receipt()
        else {
            anyhow::bail!("this prepared DiT stage requires explicit shared-weight attachment")
        };
        ensure!(
            evidence == weights.evidence()
                && evidence.weight_plan_fingerprint()
                    == plan.execution_weights().stored()?.fingerprint()?,
            "attached DiT snapshot or complete physical weight plan differs"
        );
        Ok(())
    }
    pub fn bind_refinement<'a, R: DeviceRuntime>(
        &'a self,
        session: &'a mut TensorComponentSession<R>,
        weights: &'a SharedComponentWeights<R>,
        work_host: HostMemoryAccount,
    ) -> Result<PH3ImageRefinementSession<'a, R>> {
        self.validate_attached_session(session, weights)?;
        Ok(PH3ImageRefinementSession {
            inner: dit_forward::PH3ImageTextRefinement::bind(
                &self.config,
                self.limits,
                session,
                weights,
                work_host,
            )?,
        })
    }
    pub fn velocity_layout(&self) -> Result<PH3ImageVelocityBatchLayout<'_>> {
        ensure!(
            self.phase == PH3ImageDenoiserPhase::DenoiseRefinedText,
            "only an evaluated denoising stage has velocity membership"
        );
        PH3ImageVelocityBatchLayout::new(
            &self.transformer,
            &self.velocity,
            self.limits.max_metadata_bytes,
        )
    }
    /// The engine supplies the already declared per-request times and packed
    /// positions. All large matrices are charged to its existing work account.
    pub fn prepare_inputs(
        &self,
        host: &HostMemoryAccount,
        samples: &[PH3ImageDenoiserSample<'_>],
        references: &[Option<PH3ImageDenoiserReference<'_>>],
        caches: &[PH3ImageRefinedTextCache],
        weights: &SharedComponentWeightsEvidence,
        times: &[f32],
        positions: &[[f64; 3]],
    ) -> Result<Vec<ManagedHostTensor>> {
        ensure!(
            self.phase == PH3ImageDenoiserPhase::DenoiseRefinedText,
            "install/refine stages cannot consume image/audio sampler inputs"
        );
        ensure!(
            caches.len() == self.identities.len()
                && caches
                    .iter()
                    .zip(&self.identities)
                    .all(|(c, id)| c.key().request_identity() == id
                        && c.key().matches_snapshot(weights)),
            "conditioning cache membership or installed snapshot differs"
        );
        let keys = caches.iter().map(|c| c.key().clone()).collect::<Vec<_>>();
        let samples = samples
            .iter()
            .map(|s| dit_forward::PH3ImageDiTSample {
                image: s.image,
                audio: s.audio,
            })
            .collect::<Vec<_>>();
        let references = references
            .iter()
            .map(|r| {
                r.map(|r| dit_forward::PH3ImageDiTReference {
                    latent: r.latent,
                    patch_noise: r.patch_noise,
                    augmentation: r.augmentation,
                })
            })
            .collect::<Vec<_>>();
        let basis = self.managed_basis(host, times)?;
        let (cosine, sine) = PH3ImageCheckpointRotary::new(&self.transformer, &self.source)?
            .managed_bf16_frequencies(
                host,
                positions,
                ProgramValueId::new(PH3ImageDiTFamily::COSINE_ID)?,
                ProgramValueId::new(PH3ImageDiTFamily::SINE_ID)?,
            )?;
        Ok(
            dit_forward::PH3ImageDiTInputAdapter::new(&self.config, host)?
                .prepare(
                    &samples,
                    &references,
                    dit_forward::PH3ImageDiTTextInput::RefinedText {
                        caches,
                        keys: &keys,
                    },
                    basis,
                    cosine,
                    sine,
                )?
                .inputs,
        )
    }
    fn managed_basis(&self, host: &HostMemoryAccount, times: &[f32]) -> Result<ManagedHostTensor> {
        let spec =
            &self.config.input_tensor_specs()?[&ProgramValueId::new(PH3ImageDiTFamily::BASIS_ID)?];
        ensure!(
            times.len() == usize::try_from(spec.dimensions[0])?,
            "DiT time row count differs from prepared basis"
        );
        let table = self.source.archive().tensor("adaln_t_table")?;
        let source: &[f32] = bytemuck::try_cast_slice(table.bytes())
            .map_err(|e| anyhow::anyhow!("original curve table alignment: {e}"))?;
        let mut allocation = host.allocate_zeroed(
            HostMemoryClass::InputPreprocess,
            usize::try_from(spec.byte_len()?)?,
        )?;
        let out: &mut [f32] = bytemuck::try_cast_slice_mut(allocation.as_mut())
            .map_err(|e| anyhow::anyhow!("managed curve basis alignment: {e}"))?;
        self.curve
            .write_f32(source, times, PH3ImageCurveLerpArithmetic::SeparateF32, out)?;
        Ok(ManagedHostTensor::new(
            ProgramValueId::new(PH3ImageDiTFamily::BASIS_ID)?,
            ResolvedTensorSpec::new(
                spec.dimensions.clone(),
                ElementType::F32,
                ResolvedTensorLayout::Contiguous,
            )?,
            allocation.freeze(),
        )?)
    }
}
#[derive(Clone, Copy)]
pub struct PH3ImageDenoiserSample<'a> {
    pub image: &'a ManagedHostTensor,
    pub audio: &'a ManagedHostTensor,
}
#[derive(Clone, Copy)]
pub struct PH3ImageDenoiserReference<'a> {
    pub latent: &'a ManagedHostTensor,
    pub patch_noise: Option<&'a ManagedHostTensor>,
    /// Python scalar precision; the patch adapter rounds each multiplier to F32.
    pub augmentation: f64,
}
pub struct PH3ImageRefinementSession<'a, R: DeviceRuntime> {
    inner: dit_forward::PH3ImageTextRefinement<'a, R>,
}
impl<R: DeviceRuntime> PH3ImageRefinementSession<'_, R> {
    pub fn execute_on_actor(
        &mut self,
        actor: &mut TensorSerialActor<R>,
        run: RunId,
        request: RequestIdentity,
        connector: &ManagedHostTensor,
        destinations: &[PH3ImageRefinementDestination<'_>],
    ) -> std::result::Result<Vec<PH3ImageRefinedTextCache>, PH3ImageTextRefinementError<R>> {
        self.inner
            .execute_on_actor(actor, run, request, connector, destinations)
    }
}
#[cfg(test)]
mod tests;
