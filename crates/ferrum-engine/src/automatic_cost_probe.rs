//! Product-resolved request templates for private automatic calibration.
//! These contain real API and renderer state. They grant no prepared-prefix,
//! execution, sampling or qualified-model authority.
use ferrum_interfaces::output_flow::OutputProjectionContract;
use ferrum_types::{
    ApiRequest, FerrumError, InferenceRequest, RequestId, Result,
    SloAutomaticCostProbeSamplingPresetV1,
};
use std::{num::NonZeroUsize, sync::Arc};

mod context;
pub use context::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AutomaticCostProbeOutput {
    CliText,
    ApiChatSse {
        include_usage: bool,
    },
    /// The product Completions endpoint always includes usage.
    ApiCompletionSse,
}

/// Construct at the product composition root, after applying its actual chat
/// template and the same resolved sampling defaults used by inference.
#[derive(Debug, Clone)]
pub struct AutomaticCostProbeTemplate {
    request_bytes: Arc<[u8]>,
    prompt: Arc<str>,
    output: AutomaticCostProbeOutput,
    response_model: Arc<str>,
    prompt_renderer: Option<Arc<dyn AutomaticCostProbePromptRenderer>>,
}
impl AutomaticCostProbeTemplate {
    pub fn new(request: InferenceRequest, output: AutomaticCostProbeOutput) -> Result<Self> {
        request.sampling_params.validate()?;
        if request.prompt.is_empty() || !request.stream {
            return Err(FerrumError::invalid_request(
                "automatic probe requires a nonempty rendered prompt and real streaming request",
            ));
        }
        match (output, request.api_request.as_ref()) {
            (AutomaticCostProbeOutput::CliText, None) => {}
            (AutomaticCostProbeOutput::ApiCompletionSse, Some(ApiRequest::Completion(api)))
                if api.prompt == request.prompt => {}
            (
                AutomaticCostProbeOutput::ApiChatSse { include_usage },
                Some(ApiRequest::Chat(api)),
            ) if !api.messages.is_empty()
                && api
                    .stream_options
                    .as_ref()
                    .and_then(|s| s.include_usage)
                    .unwrap_or(false)
                    == include_usage
                && request
                    .metadata
                    .get(ferrum_types::PROMPT_OPENED_REASONING_METADATA_KEY)
                    .and_then(serde_json::Value::as_bool)
                    .is_some() => {}
            _ => {
                return Err(FerrumError::invalid_request(
                    "automatic probe codec differs from original API/template metadata",
                ))
            }
        }
        // OutputPlan at actual admission remains authoritative for supported
        // native/grammar/decoder/credit bounds; this constructor does not
        // replace those checks with a probe-specific approximation.
        let response_model: Arc<str> = request.model_id.to_string().into();
        let prompt: Arc<str> = request.prompt.as_str().into();
        // These bytes are the actual retained representation, not a proxy for
        // the heap size of an also-retained serde/HashMap tree.
        let request_bytes: Arc<[u8]> = serde_json::to_vec(&request)
            .map_err(|e| FerrumError::invalid_request(format!("probe request encoding: {e}")))?
            .into();
        Ok(Self {
            request_bytes,
            prompt,
            output,
            response_model,
            prompt_renderer: None,
        })
    }
    /// Preserve the endpoint's actual public model label in its serialized
    /// envelope. It does not change the executor's internal model identity.
    pub fn with_response_model(mut self, model: String) -> Self {
        self.response_model = model.into();
        self
    }
    pub fn output(&self) -> AutomaticCostProbeOutput {
        self.output
    }
    pub fn supports_preset(&self, preset: SloAutomaticCostProbeSamplingPresetV1) -> bool {
        preset == SloAutomaticCostProbeSamplingPresetV1::Configured
            || self.output != AutomaticCostProbeOutput::ApiCompletionSse
    }
    pub fn response_model(&self) -> &str {
        &self.response_model
    }
    pub fn prompt(&self) -> &str {
        &self.prompt
    }
    pub fn serialized_request(&self) -> &[u8] {
        &self.request_bytes
    }
    /// Cold planning read; this never runs on a device dispatch or model query.
    pub fn resolved_request(&self) -> Result<InferenceRequest> {
        serde_json::from_slice(&self.request_bytes)
            .map_err(|e| FerrumError::invalid_request(format!("probe request decoding: {e}")))
    }
    /// Actual retained byte slabs plus the three Arc headers. Shared clones
    /// may be conservatively charged again; no HashMap bucket estimate exists.
    pub fn retained_payload_bytes(&self) -> Option<usize> {
        std::mem::size_of::<Self>()
            .checked_add(self.request_bytes.len())?
            .checked_add(self.prompt.len())?
            .checked_add(self.response_model.len())?
            .checked_add(match &self.prompt_renderer {
                Some(renderer) => renderer.retained_payload_bytes()?,
                None => 0,
            })?
            .checked_add(6usize.checked_mul(std::mem::size_of::<usize>())?)
    }

    /// Cold product-owned rendering capability. Instantiation and hot request
    /// paths never invoke it; frozen geometry variants do not retain it.
    pub fn with_prompt_renderer(
        mut self,
        renderer: Arc<dyn AutomaticCostProbePromptRenderer>,
    ) -> Self {
        self.prompt_renderer = Some(renderer);
        self
    }

    pub fn instantiate(
        &self,
        maximum_output: NonZeroUsize,
        seed: u64,
        preset: SloAutomaticCostProbeSamplingPresetV1,
    ) -> Result<(InferenceRequest, Arc<OutputProjectionContract>)> {
        if !self.supports_preset(preset) {
            return Err(FerrumError::unsupported(
                "Completions does not expose an ignore-EOS Length workload",
            ));
        }
        let mut request = self.resolved_request()?;
        request.id = RequestId::new();
        request.created_at = chrono::Utc::now();
        request.client_id = None;
        request.session_id = None;
        request.evidence_request = Default::default();
        request
            .metadata
            .remove(ferrum_types::DEFAULT_MAX_TOKENS_METADATA_KEY);
        let sampling = &mut request.sampling_params;
        sampling.max_tokens = maximum_output.get();
        sampling.seed = Some(seed);
        if preset == SloAutomaticCostProbeSamplingPresetV1::GreedyLength {
            sampling.temperature = 0.0;
            sampling.top_p = 1.0;
            sampling.top_k = None;
            sampling.repetition_penalty = 1.0;
            sampling.presence_penalty = 0.0;
            sampling.frequency_penalty = 0.0;
            sampling.min_p = None;
            sampling.tfs = None;
            sampling.typical_p = None;
            sampling.mirostat = None;
            sampling.stop_sequences.clear();
            request
                .metadata
                .insert("ferrum_ignore_eos".into(), true.into());
        }
        // Preserve response format, completion boundary, model protocol and
        // renderer masks. A distinct preset never removes template semantics.
        request.sampling_params.validate()?;
        let contract = match self.output {
            AutomaticCostProbeOutput::CliText => OutputProjectionContract::cli_text(),
            AutomaticCostProbeOutput::ApiChatSse { include_usage } => {
                OutputProjectionContract::chat_sse(
                    request.id.to_string(),
                    self.response_model.to_string(),
                    include_usage,
                )
            }
            AutomaticCostProbeOutput::ApiCompletionSse => {
                OutputProjectionContract::completions_sse(
                    request.id.to_string(),
                    self.response_model.to_string(),
                    true,
                )
            }
        };
        Ok((request, Arc::new(contract)))
    }
}

#[cfg(test)]
#[path = "automatic_cost_probe/tests.rs"]
mod tests;
