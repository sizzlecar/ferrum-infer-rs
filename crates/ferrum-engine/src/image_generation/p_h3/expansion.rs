//! Native prompt-expansion assets and preparation. Request rendering is CPU
//! metadata work; only an explicitly admitted phase may install or execute.
//! The image initializer must prepare its downstream geometry from the final
//! extracted text, and retain the phase until the original executor retires.

pub(crate) mod cohort;
pub(crate) mod execute;
pub(crate) mod mtp_assets;
pub(crate) mod mtp_cohort;
pub(crate) mod phase;
mod preparation;
pub(crate) mod product;
mod source_backing;
pub use source_backing::PH3ImagePromptExpansionSourceBacking;
#[cfg(test)]
mod replay_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
pub(in crate::image_generation::p_h3) use visual_assets::tests::fixture::MergedExpansionFixture;
pub(crate) mod vision;
mod visual_assets;
mod visual_inputs;
mod visual_preparation;

use ferrum_execution::{
    CompiledProgram, ExecutorBuildResult, ManagedCausalIoProfile, ProgramExecutor,
    ProgramExecutorConfig, RuntimeComposition,
};
use ferrum_models::{
    p_h3_image::{
        PH3ImagePromptExpansionGeneration, PH3ImagePromptExpansionLimits,
        PH3ImagePromptExpansionProtocol, PH3ImagePromptExpansionRequest,
    },
    vnext::{
        resolve_registered_model_from_sources, DefinedProductionModel, ManagedProductionModel,
        ManagedProductionWeightSource, ProductionModelSourceBundle,
    },
};
use ferrum_runtime::{
    DeviceRuntime, ExecutablePlanView, HostMemoryAccount, ModelArtifactSourceRole,
    NumericalProfileResolution, WeightMaterializerId, WeightMaterializerSelection,
    IDENTITY_WEIGHT_MATERIALIZER_ID,
};
use ferrum_types::{EngineConfig, FerrumError, Result, SamplingParams, TokenId};
use std::sync::Arc;

pub(crate) use preparation::ExpansionHostRequirements;
mod diagnostics;
pub(in crate::image_generation::p_h3) use diagnostics::DiagnosticContext;
pub(in crate::image_generation::p_h3) use diagnostics::VisionBindingReceipt;
pub use diagnostics::{
    PH3ImagePromptExpansionDiagnosticPolicy, PH3_IMAGE_PROMPT_EXPANSION_MAX_CAPTURE_BYTES,
};

/// Declared causal phase limits, shared by run and serve. The selected model
/// still fixes greedy sampling and its original tokenizer/template semantics.
#[derive(Clone, Debug)]
pub struct PH3ImagePromptExpansionOptions {
    pub engine: EngineConfig,
    pub max_new_tokens: u32,
    pub prefill_chunk_tokens: usize,
}

/// Reuse only an already retired, immutable visual output inside one sealed
/// expansion cohort. Each participant still binds its own causal inputs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PH3ImagePromptExpansionVisionReuse {
    #[default]
    Off,
    ExactImageWithinCohort,
}

/// Greedy speculative expansion with the checkpoint's native MTP head.
/// Disabled by default until the selected device/workload is qualified.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PH3ImagePromptExpansionMtp {
    #[default]
    Off,
    OneDraft,
}

pub(crate) struct ConfiguredTextExpansion {
    pub(crate) assets: Arc<NativeTextExpansion>,
    pub(crate) options: PH3ImagePromptExpansionOptions,
}

pub(crate) struct NativeTextExpansion {
    defined: DefinedProductionModel,
    precision: ferrum_models::vnext::qwen35::Qwen35DensePrecision,
    model: ManagedProductionModel,
    visual_model: Option<ManagedProductionModel>,
    visual_assets: Option<Arc<visual_assets::ExpansionVisualAssets>>,
    source_backing: PH3ImagePromptExpansionSourceBacking,
    source_conversion_policy: ferrum_quantization::Bf16F16ConversionPolicy,
    vision_reuse: PH3ImagePromptExpansionVisionReuse,
    mtp: PH3ImagePromptExpansionMtp,
    protocol: PH3ImagePromptExpansionProtocol,
    domain: Arc<()>,
}

/// Bounded metadata, not a device plan, Host grant, or expanded prompt. Its
/// tokenizer/parser heap and the existing executor's token metadata are not
/// represented by the transfer/backing budget below.
pub(crate) struct PreparedTextExpansion {
    domain: Arc<()>,
    prompt: String,
    width: u32,
    height: u32,
    tokens: Vec<TokenId>,
    generation: PH3ImagePromptExpansionGeneration,
    prefill_chunk_tokens: usize,
    engine: EngineConfig,
    visual: Option<visual_inputs::VisualExpansionInputs>,
    causal_inputs: Option<Arc<ferrum_execution::CausalRequestInputs>>,
    vision_binding: Option<diagnostics::VisionBindingReceipt>,
    device_timing: ferrum_runtime::DeviceTimingMode,
}

impl PreparedTextExpansion {
    pub(in crate::image_generation::p_h3) fn record_vision_binding(
        &mut self,
        receipt: diagnostics::VisionBindingReceipt,
    ) {
        self.vision_binding = Some(receipt);
    }
    #[cfg(test)]
    pub(in crate::image_generation::p_h3) fn vision_binding(
        &self,
    ) -> Option<diagnostics::VisionBindingReceipt> {
        self.vision_binding
    }
}

impl NativeTextExpansion {
    pub(crate) fn configure_mtp(&mut self, policy: PH3ImagePromptExpansionMtp) -> Result<()> {
        if self.precision == ferrum_models::vnext::qwen35::Qwen35DensePrecision::Bf16
            && policy != PH3ImagePromptExpansionMtp::Off
        {
            return Err(FerrumError::config(
                "BF16 prompt expansion currently requires MTP off",
            ));
        }
        self.mtp = policy;
        Ok(())
    }

    pub(crate) fn precision_for_engine(
        engine: &EngineConfig,
    ) -> Result<ferrum_models::vnext::qwen35::Qwen35DensePrecision> {
        use ferrum_models::vnext::qwen35::*;
        match &engine.numerical_execution {
            ferrum_types::NumericalExecutionPolicy::Auto => Ok(Qwen35DensePrecision::F16),
            ferrum_types::NumericalExecutionPolicy::Require(id)
                if id.as_str() == F16_NUMERICAL_PROFILE_ID =>
            {
                Ok(Qwen35DensePrecision::F16)
            }
            ferrum_types::NumericalExecutionPolicy::Require(id)
                if id.as_str() == BF16_NUMERICAL_PROFILE_ID =>
            {
                Ok(Qwen35DensePrecision::Bf16)
            }
            _ => Err(FerrumError::config(
                "unsupported prompt expansion numerical profile",
            )),
        }
    }
    pub(super) fn is_bf16(&self) -> bool {
        self.precision == ferrum_models::vnext::qwen35::Qwen35DensePrecision::Bf16
    }
    fn numerical_profile(&self, visual: bool) -> &'static str {
        use ferrum_models::vnext::qwen35::*;
        match (self.is_bf16(), visual) {
            (false, false) => F16_NUMERICAL_PROFILE_ID,
            (false, true) => F16_VISUAL_PROMPT_NUMERICAL_PROFILE_ID,
            (true, false) => BF16_NUMERICAL_PROFILE_ID,
            (true, true) => BF16_VISUAL_PROMPT_NUMERICAL_PROFILE_ID,
        }
    }
    fn managed_io_profile(&self) -> ManagedCausalIoProfile {
        if self.is_bf16() {
            ManagedCausalIoProfile::ManagedGreedyBf16V1
        } else {
            ManagedCausalIoProfile::ManagedGreedyF16V1
        }
    }
    fn kv_storage(&self) -> ferrum_types::KvStorageFormat {
        if self.is_bf16() {
            ferrum_types::KvStorageFormat::Bf16
        } else {
            ferrum_types::KvStorageFormat::F16
        }
    }
    pub(crate) fn configure_vision_reuse(&mut self, policy: PH3ImagePromptExpansionVisionReuse) {
        self.vision_reuse = policy;
    }

    pub(crate) fn vision_reuse(&self) -> PH3ImagePromptExpansionVisionReuse {
        self.vision_reuse
    }
    pub(crate) fn configure_process_mapped_weights(
        &mut self,
        directory: std::path::PathBuf,
        limits: ferrum_quantization::ProcessMappedDenseLimits,
        upload_policy: ferrum_quantization::ProcessMappedDenseUploadPolicy,
    ) -> Result<()> {
        if self.source_backing != PH3ImagePromptExpansionSourceBacking::Fresh {
            return Err(FerrumError::config(
                "installation-local source backing cannot be combined with process-mapped weights",
            ));
        }
        self.model
            .configure_process_mapped_weights_with_upload_policy(
                directory,
                limits,
                upload_policy,
            )?;
        if let Some(visual) = &mut self.visual_model {
            visual.share_process_mapped_weights_from(&self.model)?;
        }
        Ok(())
    }

    pub(crate) fn trace_identity(&self) -> anyhow::Result<serde_json::Value> {
        let visual = self.visual_assets.as_ref().map(|assets| {
            serde_json::json!({
                "sources": assets.source_identity(),
                "fingerprint": assets.fingerprint(),
                "profile": assets.profile(),
            })
        });
        Ok(serde_json::json!({
            "sources": self.model.prepared().sources().resolved_sources(),
            "family_fingerprint": self.model.prepared().family().fingerprint()?,
            "process_mapped_execution_weights": self.model.process_mapped_weights_enabled(),
            "process_mapped_upload_policy": self.model.process_mapped_upload_policy(),
            "source_backing": self.source_backing,
            "source_conversion_workers": self.source_conversion_policy.maximum_workers().get(),
            "vision_reuse": self.vision_reuse,
            "mtp": self.mtp,
            "visual": visual,
            "visual_causal_family_fingerprint": self.visual_model.as_ref()
                .map(|model| model.prepared().family().fingerprint()).transpose()?,
        }))
    }
    pub(super) fn family(&self) -> &ferrum_runtime::PreparedModelFamily {
        self.model.prepared().family()
    }
    pub(crate) fn from_sources(
        sources: Arc<ProductionModelSourceBundle>,
        limits: PH3ImagePromptExpansionLimits,
    ) -> Result<Self> {
        Self::from_sources_with_precision(
            sources,
            limits,
            ferrum_models::vnext::qwen35::Qwen35DensePrecision::F16,
        )
    }

    pub(crate) fn from_sources_with_precision(
        sources: Arc<ProductionModelSourceBundle>,
        limits: PH3ImagePromptExpansionLimits,
        precision: ferrum_models::vnext::qwen35::Qwen35DensePrecision,
    ) -> Result<Self> {
        let config = sources.tokenizer_config_json().ok_or_else(|| {
            FerrumError::model("prompt expansion requires its original tokenizer config")
        })?;
        let template = sources.chat_template_jinja().ok_or_else(|| {
            FerrumError::model("prompt expansion requires its original processor Jinja template")
        })?;
        let protocol = PH3ImagePromptExpansionProtocol::from_source_bytes(
            sources.tokenizer_json(),
            config,
            template,
            limits,
        )
        .map_err(model_error)?;
        for (name, hash) in [
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
        ]
        .into_iter()
        .zip(protocol.source_sha256())
        {
            if sources
                .fingerprint(ModelArtifactSourceRole::Tokenizer, name)
                .is_none_or(|evidence| &evidence.sha256 != hash)
            {
                return Err(FerrumError::model(
                    "prompt protocol bytes differ from the frozen tokenizer source",
                ));
            }
        }
        let defined = match precision {
            ferrum_models::vnext::qwen35::Qwen35DensePrecision::F16 => {
                resolve_registered_model_from_sources(&sources)?
                    .into_required()?
                    .define_from_sources(sources)?
            }
            ferrum_models::vnext::qwen35::Qwen35DensePrecision::Bf16 => {
                ferrum_models::vnext::qwen35::define_from_sources_with_dense_precision(
                    sources, precision,
                )?
            }
        };
        let profile = match precision {
            ferrum_models::vnext::qwen35::Qwen35DensePrecision::F16 => {
                ferrum_models::vnext::qwen35::F16_NUMERICAL_PROFILE_ID
            }
            ferrum_models::vnext::qwen35::Qwen35DensePrecision::Bf16 => {
                ferrum_models::vnext::qwen35::BF16_NUMERICAL_PROFILE_ID
            }
        }
        .parse()
        .map_err(model_error)?;
        let prepared = Arc::new(defined.prepare(&profile)?);
        if [protocol.eos_token_id(), protocol.pad_token_id()]
            .into_iter()
            .any(|id| id as usize >= prepared.descriptor().vocabulary_size())
        {
            return Err(FerrumError::model(
                "prompt protocol token IDs exceed the prepared vocabulary",
            ));
        }
        Ok(Self {
            defined,
            precision,
            model: prepared.managed_safetensors()?,
            visual_model: None,
            visual_assets: None,
            source_backing: PH3ImagePromptExpansionSourceBacking::default(),
            source_conversion_policy: ferrum_quantization::Bf16F16ConversionPolicy::default(),
            vision_reuse: PH3ImagePromptExpansionVisionReuse::default(),
            mtp: PH3ImagePromptExpansionMtp::default(),
            protocol,
            domain: Arc::new(()),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare_text(
        &self,
        prompt: &str,
        width: u32,
        height: u32,
        max_new_tokens: u32,
        prefill_chunk_tokens: usize,
        mut engine: EngineConfig,
    ) -> Result<PreparedTextExpansion> {
        if prefill_chunk_tokens == 0 {
            return Err(FerrumError::config("prompt prefill chunk must be positive"));
        }
        let request = PH3ImagePromptExpansionRequest::text_to_image(prompt, width, height)
            .and_then(|r| r.with_max_new_tokens(max_new_tokens))
            .map_err(model_error)?;
        let rendered = self.protocol.render(&request).map_err(model_error)?;
        let ids = self
            .protocol
            .encode_text_input(&rendered)
            .map_err(model_error)?;
        let generation = self.protocol.generation(&request).map_err(model_error)?;
        let context = ids
            .len()
            .checked_add(generation.max_new_tokens as usize)
            .ok_or_else(|| FerrumError::config("prompt expansion context overflows"))?;
        if ids.is_empty()
            || context > self.model.prepared().descriptor().maximum_sequence_tokens()
            || engine
                .runtime
                .max_model_len
                .is_some_and(|ceiling| context > ceiling)
            || ids
                .iter()
                .any(|&id| id as usize >= self.model.prepared().descriptor().vocabulary_size())
        {
            return Err(FerrumError::request_validation(
                "prompt expansion exceeds its declared context or vocabulary",
            ));
        }
        engine.runtime.max_model_len = Some(context);
        engine.scheduler.max_running_requests = 1;
        engine.scheduler.max_waiting_requests = 1;
        engine.batching.max_num_batched_tokens = prefill_chunk_tokens.min(ids.len());
        // H3 remains eager by default; only the explicit decode-only policy opts in.
        engine.backend.enable_reusable_execution &= engine.backend.reusable_execution_capture.scope
            == ferrum_types::ReusableExecutionCaptureScope::DecodeOnly;
        engine.kv_cache.dtype = if self.is_bf16() {
            ferrum_types::KvCacheDtype::Bf16
        } else {
            ferrum_types::KvCacheDtype::Fp16
        };
        engine.numerical_execution = self.numerical_profile(false).parse().map_err(model_error)?;
        engine.sampling.default_params = SamplingParams::greedy();
        engine.sampling.default_params.max_tokens = generation.max_new_tokens as usize;
        Ok(PreparedTextExpansion {
            domain: Arc::clone(&self.domain),
            prompt: prompt.to_owned(),
            width,
            height,
            tokens: ids.into_iter().map(TokenId::new).collect(),
            generation,
            prefill_chunk_tokens: engine.batching.max_num_batched_tokens,
            engine,
            visual: None,
            causal_inputs: None,
            vision_binding: None,
            device_timing: ferrum_runtime::DeviceTimingMode::Off,
        })
    }

    fn validate_preparation(&self, prepared: &PreparedTextExpansion) -> Result<()> {
        if !Arc::ptr_eq(&self.domain, &prepared.domain) {
            return Err(FerrumError::request_validation(
                "prompt preparation belongs to another source owner",
            ));
        }
        Ok(())
    }

    fn selected_model(&self, prepared: &PreparedTextExpansion) -> Result<&ManagedProductionModel> {
        if prepared.visual.is_some() {
            self.visual_model.as_ref().ok_or_else(|| {
                FerrumError::unsupported(
                    "image-conditioned prompt expansion model was not prepared",
                )
            })
        } else {
            Ok(&self.model)
        }
    }

    fn compile<'a, R: DeviceRuntime>(
        &'a self,
        composition: &'a RuntimeComposition<R>,
        source: &'a ManagedProductionWeightSource<'a>,
        prepared: &PreparedTextExpansion,
    ) -> Result<CompiledProgram<'a, R>> {
        self.compile_with_engine(composition, source, prepared, &prepared.engine)
    }

    fn compile_with_engine<'a, R: DeviceRuntime>(
        &'a self,
        composition: &'a RuntimeComposition<R>,
        source: &'a ManagedProductionWeightSource<'a>,
        prepared: &PreparedTextExpansion,
        engine: &EngineConfig,
    ) -> Result<CompiledProgram<'a, R>> {
        self.validate_preparation(prepared)?;
        let info = self
            .selected_model(prepared)?
            .prepared()
            .model_info(engine.model.model_id.clone(), engine.backend.device.clone());
        let config =
            ProgramExecutorConfig::from_engine_config(engine, &info, composition.runtime())?;
        let materializer = WeightMaterializerSelection::exact(
            WeightMaterializerId::new(IDENTITY_WEIGHT_MATERIALIZER_ID).map_err(model_error)?,
        );
        if let Some(visual) = &prepared.visual {
            composition
                .compile_model_with_causal_inputs(
                    source.execution_input()?,
                    info,
                    engine,
                    config,
                    materializer,
                    visual.schema(),
                )
                .map_err(FerrumError::from)
        } else {
            composition
                .compile_model(
                    source.execution_input()?,
                    info,
                    engine,
                    config,
                    materializer,
                )
                .map_err(FerrumError::from)
        }
    }

    /// The caller must supply the actual admitted phase account and retain its
    /// scope plus construction cleanup until the original executor has retired.
    /// No scope, runtime, second ledger, or hidden model cache is created here.
    pub(crate) fn initialize<R: DeviceRuntime>(
        &self,
        composition: &RuntimeComposition<R>,
        prepared: &PreparedTextExpansion,
        phase_host: &HostMemoryAccount,
    ) -> ExecutorBuildResult<ProgramExecutor<R>> {
        self.validate_bound_inputs(prepared)?;
        let source = self.selected_model(prepared)?.weight_source(phase_host);
        let compiled = self.compile(composition, &source, prepared)?;
        let _backing = compiled.prepare_static_host_backing()?;
        self.prepare_static_source(
            &source,
            compiled.compilation().executable().execution_plan(),
        )?;
        compiled.initialize_with_managed_io(
            phase_host,
            self.managed_io_profile(),
            |policy, catalog, compilation| self.resolve(prepared, policy, catalog, compilation),
        )
    }

    /// Transfer the actual phase and materializer lease before the original
    /// constructor can acquire native resources. Its own cleanup owns them.
    pub(crate) fn initialize_owned<R: DeviceRuntime>(
        &self,
        composition: &RuntimeComposition<R>,
        prepared: &PreparedTextExpansion,
        phase: ferrum_runtime::HostMemoryScope,
    ) -> ExecutorBuildResult<ferrum_execution::executor_lifecycle::ExecutorOwner> {
        self.initialize_owned_with_engine(composition, prepared, &prepared.engine, phase)
    }

    fn initialize_owned_with_engine<R: DeviceRuntime>(
        &self,
        composition: &RuntimeComposition<R>,
        prepared: &PreparedTextExpansion,
        engine: &EngineConfig,
        phase: ferrum_runtime::HostMemoryScope,
    ) -> ExecutorBuildResult<ferrum_execution::executor_lifecycle::ExecutorOwner> {
        let host = phase.account();
        self.validate_bound_inputs(prepared)?;
        let source = self.selected_model(prepared)?.weight_source(&host);
        let compiled = self.compile_with_engine(composition, &source, prepared, engine)?;
        let backing = compiled.prepare_static_host_backing()?;
        self.prepare_static_source(
            &source,
            compiled.compilation().executable().execution_plan(),
        )?;
        let owner = compiled.initialize_with_managed_io_retained(
            &host,
            self.managed_io_profile(),
            (phase, backing),
            |policy, catalog, compilation| {
                self.resolve_with_engine(prepared, engine, policy, catalog, compilation)
            },
        )?;
        prepare_owned_startup(&owner)?;
        Ok(owner)
    }

    fn resolve(
        &self,
        prepared: &PreparedTextExpansion,
        policy: &ferrum_runtime::ResolvedRuntimePolicy,
        catalog: &ferrum_runtime::CapabilityCatalog,
        compilation: &ferrum_runtime::ProgramPlanCompilation,
    ) -> Result<ferrum_runtime::ResolvedModelPlan> {
        self.resolve_with_engine(prepared, &prepared.engine, policy, catalog, compilation)
    }

    fn resolve_with_engine(
        &self,
        prepared: &PreparedTextExpansion,
        engine: &EngineConfig,
        policy: &ferrum_runtime::ResolvedRuntimePolicy,
        catalog: &ferrum_runtime::CapabilityCatalog,
        compilation: &ferrum_runtime::ProgramPlanCompilation,
    ) -> Result<ferrum_runtime::ResolvedModelPlan> {
        let numerical = NumericalProfileResolution::from_static_plan(
            engine.numerical_execution.clone(),
            self.kv_storage(),
            self.defined.definition(),
            self.selected_model(prepared)?.prepared().family(),
            catalog,
            policy,
            compilation.executable().execution_plan(),
            vec![],
        )
        .map_err(model_error)?;
        crate::product_composition::resolve_model_plan(
            engine,
            self.selected_model(prepared)?.prepared(),
            catalog,
            policy,
            compilation,
            &numerical,
        )
    }
    pub(crate) fn extract(
        &self,
        prepared: &PreparedTextExpansion,
        ids: &[u32],
        finish: ferrum_types::FinishReason,
    ) -> Result<String> {
        self.validate_preparation(prepared)?;
        let request = if let Some(visual) = &prepared.visual {
            PH3ImagePromptExpansionRequest::edit(
                &prepared.prompt,
                visual.tokens().image_runs().len() as u8,
            )
        } else {
            PH3ImagePromptExpansionRequest::text_to_image(
                &prepared.prompt,
                prepared.width,
                prepared.height,
            )
        }
        .and_then(|r| r.with_max_new_tokens(prepared.generation.max_new_tokens))
        .map_err(model_error)?;
        self.protocol
            .extract(&request, ids, finish)
            .map_err(model_error)
    }
}

fn prepare_owned_startup(
    owner: &ferrum_execution::executor_lifecycle::ExecutorOwner,
) -> ExecutorBuildResult<()> {
    let scope = owner.scope().map_err(|error| {
        ferrum_execution::ExecutorBuildError::with_cleanup(error, owner.begin_shutdown())
    })?;
    // H3 runs synchronous native phases on its existing worker. On-demand
    // startup configures the original bounded catalog before any admission;
    // it does not execute synthetic prefill or decode work.
    let result = futures::executor::block_on(tokio::task::unconstrained(scope.prepare_startup()));
    drop(scope);
    result.map_err(|error| {
        ferrum_execution::ExecutorBuildError::with_cleanup(error, owner.begin_shutdown())
    })
}

fn model_error(error: impl std::fmt::Display) -> FerrumError {
    FerrumError::model(error.to_string())
}
