//! Image-conditioned causal preparation. This is usable only after the caller
//! supplies actual HF visual output; it never substitutes image-pad embeddings.
use super::*;
use ferrum_execution::TensorExecutionOutput;
use ferrum_models::{qwen35_config::Qwen35TextConfig, qwen3_vl_vision::Qwen3VLVisionGrid};

impl NativeTextExpansion {
    /// Explicitly enable the separate visual causal profile on the same
    /// already verified source mapping.
    pub(crate) fn with_visual_prompt(mut self) -> Result<Self> {
        let profile = self.numerical_profile(true).parse().map_err(model_error)?;
        self.visual_model = Some(
            self.model
                .with_prepared(Arc::new(self.defined.prepare(&profile)?))?,
        );
        Ok(self)
    }

    pub(crate) fn with_visual_assets(
        self,
        processor: ferrum_models::qwen3_vl_vision::hf::Qwen3VLVisionHfProcessor,
        limits: ferrum_models::qwen3_vl_vision::hf::Qwen3VLVisionHfLimits,
        profile: ferrum_runtime::NumericalProfileId,
    ) -> Result<Self> {
        let mut assets = self.with_visual_prompt()?;
        assets.visual_assets = Some(Arc::new(
            visual_assets::ExpansionVisualAssets::from_processor_with_profile(
                &assets.model,
                processor,
                limits,
                profile,
            )
            .map_err(model_error)?,
        ));
        Ok(assets)
    }

    pub(crate) fn visual_assets(&self) -> Option<&Arc<visual_assets::ExpansionVisualAssets>> {
        self.visual_assets.as_ref()
    }

    pub(crate) fn configure_visual_profile(
        &mut self,
        profile: ferrum_runtime::NumericalProfileId,
    ) -> Result<()> {
        let Some(previous) = &self.visual_assets else {
            return Ok(());
        };
        use ferrum_models::qwen3_vl_vision::hf::Qwen3VLVisionHfFamily;
        if !matches!(
            previous.profile().as_str(),
            Qwen3VLVisionHfFamily::CUDA_PROFILE_ID
                | Qwen3VLVisionHfFamily::CUDA_FUSED_F32_PROFILE_ID
        ) {
            return Err(FerrumError::config(
                "CUDA attention selection requires an existing CUDA HF vision profile",
            ));
        }
        if previous.profile() == &profile {
            return Ok(());
        }
        self.visual_assets = Some(Arc::new(
            visual_assets::ExpansionVisualAssets::from_processor_with_profile(
                &self.model,
                previous.processor().clone(),
                previous.limits(),
                profile,
            )
            .map_err(model_error)?,
        ));
        Ok(())
    }

    pub(crate) fn visual_causal_family(&self) -> Option<&ferrum_runtime::PreparedModelFamily> {
        self.visual_model
            .as_ref()
            .map(|model| model.prepared().family())
    }

    pub(crate) fn vision_family(&self) -> Option<&ferrum_runtime::PreparedModelFamily> {
        self.visual_assets
            .as_ref()
            .map(|assets| assets.minimal_family())
    }

    pub(crate) fn prepare_edit(
        &self,
        prompt: &str,
        dimensions: (u32, u32),
        grid: &Qwen3VLVisionGrid,
        spatial_merge_size: u64,
        options: &PH3ImagePromptExpansionOptions,
    ) -> Result<PreparedTextExpansion> {
        self.prepare_edits(
            prompt,
            dimensions,
            std::slice::from_ref(grid),
            spatial_merge_size,
            options,
        )
    }
    pub(crate) fn prepare_edits(
        &self,
        prompt: &str,
        dimensions: (u32, u32),
        grids: &[Qwen3VLVisionGrid],
        spatial_merge_size: u64,
        options: &PH3ImagePromptExpansionOptions,
    ) -> Result<PreparedTextExpansion> {
        let model = self.visual_model.as_ref().ok_or_else(|| {
            FerrumError::unsupported(
                "image-conditioned prompt expansion requires its declared visual profile",
            )
        })?;
        if options.prefill_chunk_tokens == 0 {
            return Err(FerrumError::config("prompt prefill chunk must be positive"));
        }
        let count = u8::try_from(grids.len()).map_err(model_error)?;
        let request = PH3ImagePromptExpansionRequest::edit(prompt, count)
            .and_then(|r| r.with_max_new_tokens(options.max_new_tokens))
            .map_err(model_error)?;
        let rendered = self.protocol.render(&request).map_err(model_error)?;
        let tokens = self
            .protocol
            .encode_image_input(&rendered, grids, spatial_merge_size)
            .map_err(model_error)?;
        let generation = self.protocol.generation(&request).map_err(model_error)?;
        let raw =
            serde_json::from_slice(self.defined.sources().config_json()).map_err(model_error)?;
        let text = Qwen35TextConfig::from_hf_config_value(&raw).map_err(model_error)?;
        let visual =
            visual_inputs::VisualExpansionInputs::with_precision(tokens, &text, self.precision)?;
        let context = visual.tokens().maximum_tokens();
        let ids = visual.tokens().tokens();
        let mut engine = options.engine.clone();
        if ids.is_empty()
            || context > model.prepared().descriptor().maximum_sequence_tokens()
            || engine
                .runtime
                .max_model_len
                .is_some_and(|ceiling| context > ceiling)
            || ids
                .iter()
                .any(|&id| id as usize >= model.prepared().descriptor().vocabulary_size())
        {
            return Err(FerrumError::request_validation(
                "visual prompt expansion exceeds its declared context or vocabulary",
            ));
        }
        engine.runtime.max_model_len = Some(context);
        engine.scheduler.max_running_requests = 1;
        engine.scheduler.max_waiting_requests = 1;
        engine.batching.max_num_batched_tokens = options.prefill_chunk_tokens.min(ids.len());
        // Keep the text and visual paths on the same explicit decode-only
        // policy. Core still requires managed inputs and OnDemand preparation;
        // visual replacement and explicit RoPE providers remain eager boundaries.
        engine.backend.enable_reusable_execution &= engine.backend.reusable_execution_capture.scope
            == ferrum_types::ReusableExecutionCaptureScope::DecodeOnly;
        engine.kv_cache.dtype = if self.is_bf16() {
            ferrum_types::KvCacheDtype::Bf16
        } else {
            ferrum_types::KvCacheDtype::Fp16
        };
        engine.numerical_execution = self.numerical_profile(true).parse().map_err(model_error)?;
        engine.sampling.default_params = SamplingParams::greedy();
        engine.sampling.default_params.max_tokens = generation.max_new_tokens as usize;
        Ok(PreparedTextExpansion {
            domain: Arc::clone(&self.domain),
            prompt: prompt.to_owned(),
            width: dimensions.0,
            height: dimensions.1,
            tokens: ids.iter().copied().map(TokenId::new).collect(),
            generation,
            prefill_chunk_tokens: engine.batching.max_num_batched_tokens,
            engine,
            visual: Some(visual),
            causal_inputs: None,
            vision_binding: None,
            device_timing: ferrum_runtime::DeviceTimingMode::Off,
        })
    }

    /// Called with actual main-only vision results on the admitted phase account.
    /// The resulting original owner is immutable through every admission retry.
    pub(crate) fn bind_image_outputs(
        &self,
        prepared: &mut PreparedTextExpansion,
        phase_host: &HostMemoryAccount,
        outputs: &[TensorExecutionOutput],
    ) -> Result<()> {
        self.validate_preparation(prepared)?;
        if prepared.causal_inputs.is_some() {
            return Err(FerrumError::request_validation(
                "visual expansion inputs were already bound",
            ));
        }
        prepared.causal_inputs = Some(
            prepared
                .visual
                .as_ref()
                .ok_or_else(|| {
                    FerrumError::request_validation("text expansion cannot receive visual outputs")
                })?
                .bind_outputs(phase_host, outputs)?,
        );
        Ok(())
    }

    pub(super) fn validate_bound_inputs(&self, prepared: &PreparedTextExpansion) -> Result<()> {
        self.validate_preparation(prepared)?;
        match (&prepared.visual, &prepared.causal_inputs) {
            (None, None) => Ok(()),
            (Some(plan), Some(inputs)) if inputs.schema() == plan.schema()
                && inputs.maximum_tokens() == plan.tokens().maximum_tokens() => Ok(()),
            _ => Err(FerrumError::request_validation("visual expansion requires its actual immutable feature and position inputs before installation")),
        }
    }
}
