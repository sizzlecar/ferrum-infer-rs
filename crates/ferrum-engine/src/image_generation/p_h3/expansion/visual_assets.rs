//! Immutable visual assets for the original prompt-upsampler source. Validate
//! the source once; request dimensions only rebuild plan/activation metadata.
use anyhow::{ensure, Result};
#[cfg(test)]
use ferrum_models::qwen3_vl_vision::hf::Qwen3VLVisionHfProcessorLimits;
use ferrum_models::{
    qwen3_vl_vision::{
        hf::{
            Qwen3VLVisionHfConfig, Qwen3VLVisionHfFamily, Qwen3VLVisionHfImagePlan,
            Qwen3VLVisionHfLimits, Qwen3VLVisionHfProcessor, Qwen3VLVisionHfWeightAdapter,
        },
        Qwen3VLVisionArchitecture, Qwen3VLVisionGrid,
    },
    vnext::ManagedProductionModel,
};
use ferrum_quantization::comfy_convrot::ComfyCanonicalWeightSource;
use ferrum_runtime::{
    ModelFamilyRegistration, NumericalProfileId, PreparedModelFamily, ResolvedModelSources,
    TypedFamilyRegistration,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::sync::Arc;

/// Semantic/tokenizer/weight roles retain the original frozen evidence.
/// Processor bytes are a separate artifact; their SHA does not overwrite any
/// of those roles or infer a path from a model/service name.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub(crate) struct ExpansionVisualSourceIdentity {
    pub original_model: ResolvedModelSources,
    pub processor_sha256: String,
}

pub(crate) struct ExpansionVisualAssets {
    processor: Qwen3VLVisionHfProcessor,
    architecture: Qwen3VLVisionArchitecture,
    weights: Arc<Qwen3VLVisionHfWeightAdapter>,
    limits: Qwen3VLVisionHfLimits,
    profile: NumericalProfileId,
    // Retain the validated family even when this build does not run CUDA preflight.
    _minimal_family: PreparedModelFamily,
    identity: ExpansionVisualSourceIdentity,
    fingerprint: String,
}
impl ExpansionVisualAssets {
    #[cfg(test)]
    pub(crate) fn from_processor(
        model: &ManagedProductionModel,
        processor: Qwen3VLVisionHfProcessor,
        limits: Qwen3VLVisionHfLimits,
    ) -> Result<Self> {
        Self::from_processor_with_profile(
            model,
            processor,
            limits,
            NumericalProfileId::new(Qwen3VLVisionHfFamily::CUDA_PROFILE_ID)
                .map_err(anyhow::Error::msg)?,
        )
    }
    pub(crate) fn from_processor_with_profile(
        model: &ManagedProductionModel,
        processor: Qwen3VLVisionHfProcessor,
        limits: Qwen3VLVisionHfLimits,
        profile: NumericalProfileId,
    ) -> Result<Self> {
        let raw = serde_json::from_slice(model.prepared().sources().config_json())?;
        let architecture = Qwen3VLVisionArchitecture::from_qwen35_hf_value(&raw)?;
        // Check the actual language declaration, rather than a model-name gate.
        let text = ferrum_models::qwen35_config::Qwen35TextConfig::from_hf_config_value(&raw)
            .map_err(anyhow::Error::msg)?;
        ensure!(
            architecture.output_size() == text.hidden_size as u64,
            "prompt visual merger width differs from its original causal embedding width"
        );
        let merge = architecture.merge_size();
        let minimal = Qwen3VLVisionHfConfig::new(
            architecture.clone(),
            Qwen3VLVisionGrid::new(1, merge, merge)?,
            &processor,
        )?;
        let source = Arc::new(ComfyCanonicalWeightSource::from_shared_archive(
            model.shared_archive(),
        )?);
        let minimal_family = TypedFamilyRegistration::new(Qwen3VLVisionHfFamily::new(limits)?)
            .prepare_with_profile(&serde_json::to_value(&minimal)?, &profile)?;
        let weights = Arc::new(Qwen3VLVisionHfWeightAdapter::new(&minimal, source, limits)?);
        let identity = ExpansionVisualSourceIdentity {
            original_model: model.prepared().sources().resolved_sources().clone(),
            processor_sha256: processor.source_sha256().to_owned(),
        };
        let fingerprint = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&(&identity, &profile))?)
        );
        Ok(Self {
            processor,
            architecture,
            weights,
            limits,
            profile,
            _minimal_family: minimal_family,
            identity,
            fingerprint,
        })
    }
    pub(crate) fn processor(&self) -> &Qwen3VLVisionHfProcessor {
        &self.processor
    }
    #[cfg(test)]
    pub(crate) fn architecture(&self) -> &Qwen3VLVisionArchitecture {
        &self.architecture
    }
    pub(crate) fn source_identity(&self) -> &ExpansionVisualSourceIdentity {
        &self.identity
    }
    pub(crate) fn fingerprint(&self) -> &str {
        &self.fingerprint
    }
    pub(crate) fn limits(&self) -> Qwen3VLVisionHfLimits {
        self.limits
    }
    pub(crate) fn profile(&self) -> &NumericalProfileId {
        &self.profile
    }
    pub(super) fn weights(&self) -> &Arc<Qwen3VLVisionHfWeightAdapter> {
        &self.weights
    }
    #[cfg(test)]
    pub(crate) fn stored_bytes(&self) -> Result<u64> {
        self.weights.stored_bytes()
    }
    pub(crate) fn plan(
        &self,
        source_width: u32,
        source_height: u32,
    ) -> Result<Qwen3VLVisionHfImagePlan> {
        Qwen3VLVisionHfImagePlan::new(
            self.architecture.clone(),
            source_width,
            source_height,
            &self.processor,
        )
    }
    /// Actual complete main-only program and original Stored schema at one
    /// merge cell. This is for provider preflight, not an execution receipt.
    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn minimal_family(&self) -> &PreparedModelFamily {
        &self._minimal_family
    }
}

#[cfg(test)]
pub(super) mod tests;
