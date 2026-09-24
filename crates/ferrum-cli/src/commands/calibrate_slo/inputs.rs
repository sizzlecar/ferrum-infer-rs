use super::*;
use ferrum_interfaces::output_flow::OutputProjectionContract;
use ferrum_server::chat_template::ModelChatTemplate;
use ferrum_types::{InferenceRequest, ModelId};
use sha2::{Digest, Sha256};

pub(super) enum PreparedInputs {
    Rendered,
    ShareGpt {
        recovered: sharegpt::Recovered,
        template: ModelChatTemplate,
        policy: sharegpt::RequestPolicy,
    },
}

impl PreparedInputs {
    /// The exact HTTP body before product template rendering. Reference output
    /// overrides are explicit and are never used by the HTTP input exporter.
    fn chat_body(
        prompt: &sharegpt::RecoveredPrompt,
        policy: &sharegpt::RequestPolicy,
        reference_output: Option<(usize, bool)>,
    ) -> serde_json::Value {
        let (maximum, ignore_eos) = reference_output.unwrap_or((
            prompt.sample.requested_output_tokens as usize,
            policy.ignore_eos,
        ));
        super::super::chat_request::chat_completion_body(
            &policy.requested_model_name,
            &prompt.text,
            maximum,
            ignore_eos,
            policy.enable_thinking,
            policy.reasoning_effort,
            policy.sampling,
        )
    }

    pub(super) fn original_http_body(&self, index: usize) -> Result<serde_json::Value> {
        let Self::ShareGpt {
            recovered, policy, ..
        } = self
        else {
            return Err(FerrumError::config(
                "HTTP input export requires frozen ShareGPT Chat inputs",
            ));
        };
        let prompt = recovered.prompts.get(index).ok_or_else(|| {
            FerrumError::config("HTTP input export refers to an absent frozen sample")
        })?;
        Ok(Self::chat_body(prompt, policy, None))
    }
    pub(super) fn prepare(
        manifest: &manifest::Manifest,
        template: ModelChatTemplate,
    ) -> Result<Self> {
        let Some(source) = &manifest.sharegpt else {
            return Ok(Self::Rendered);
        };
        let recovered = sharegpt::load(source)?;
        if manifest
            .cohorts()
            .flat_map(|case| &case.prompts)
            .any(|&index| index >= recovered.prompts.len())
        {
            return Err(FerrumError::config(
                "calibration cohort references an absent frozen sample",
            ));
        }
        Ok(Self::ShareGpt {
            recovered,
            template,
            policy: source.request_policy.clone(),
        })
    }

    pub(super) fn provenance(&self) -> serde_json::Value {
        match self {
            Self::Rendered => serde_json::json!({"kind":"manifest_rendered_input_v1"}),
            Self::ShareGpt { recovered, .. } => recovered.provenance.clone(),
        }
    }

    pub(super) fn output_contract(
        &self,
        codec: manifest::Codec,
        request: &InferenceRequest,
    ) -> OutputProjectionContract {
        let wire_model = match self {
            Self::Rendered => request.model_id.0.as_str(),
            Self::ShareGpt { policy, .. } => policy.requested_model_name.as_str(),
        };
        match codec {
            manifest::Codec::CliText => OutputProjectionContract::cli_text(),
            manifest::Codec::CompletionsSse { include_usage } => {
                OutputProjectionContract::completions_sse(
                    request.id.to_string(),
                    wire_model.to_owned(),
                    include_usage,
                )
            }
            manifest::Codec::ChatSse { include_usage } => OutputProjectionContract::chat_sse(
                request.id.to_string(),
                wire_model.to_owned(),
                include_usage,
            ),
        }
    }

    pub(super) fn request(
        &self,
        manifest: &manifest::Manifest,
        index: usize,
        model_id: &ModelId,
        output_protocol: ferrum_types::ModelOutputProtocol,
    ) -> Result<(InferenceRequest, serde_json::Value)> {
        self.build_request(manifest, index, model_id, output_protocol, None)
    }

    pub(super) fn count(&self, manifest: &manifest::Manifest) -> usize {
        match self {
            Self::Rendered => manifest.prompts.len(),
            Self::ShareGpt { recovered, .. } => recovered.prompts.len(),
        }
    }

    /// The phase is typed so training/heldout cannot accidentally inherit a reference override.
    pub(super) fn request_for_phase(
        &self,
        manifest: &manifest::Manifest,
        index: usize,
        model_id: &ModelId,
        output_protocol: ferrum_types::ModelOutputProtocol,
        phase: report::Phase,
    ) -> Result<(InferenceRequest, serde_json::Value)> {
        let policy = manifest
            .reference
            .as_ref()
            .map(|config| config.request_policy)
            .unwrap_or_default();
        if policy.independent(phase) {
            self.reference_request(manifest, index, model_id, output_protocol, policy)
        } else {
            self.request(manifest, index, model_id, output_protocol)
        }
    }

    fn reference_request(
        &self,
        manifest: &manifest::Manifest,
        index: usize,
        model_id: &ModelId,
        output_protocol: ferrum_types::ModelOutputProtocol,
        policy: reference::ReferenceRequestPolicy,
    ) -> Result<(InferenceRequest, serde_json::Value)> {
        policy.validate()?;
        let reference::ReferenceRequestPolicy::FixedReferenceOutput { max_tokens, eos } = policy
        else {
            return self.request(manifest, index, model_id, output_protocol);
        };
        // Construct both through the same product factory. Only the new reference
        // request is submitted; construction is not evidence the source ran.
        let (original, source) = self.request(manifest, index, model_id, output_protocol)?;
        let (request, actual) = self.build_request(
            manifest,
            index,
            model_id,
            output_protocol,
            Some((max_tokens.get(), eos.ignore())),
        )?;
        if request.prompt != original.prompt || request.model_id != original.model_id {
            return Err(FerrumError::config(
                "independent reference policy changed the actual rendered input",
            ));
        }
        let evidence = serde_json::json!({
            "kind":"independent_reference_request_v1",
            "source_index":index,
            "source_request":source,
            "reference_request":actual,
            "source_maximum_output_tokens":original.sampling_params.max_tokens,
            "reference_maximum_output_tokens":request.sampling_params.max_tokens,
            "reference_eos":eos,
            "rendered_input_equal_to_source":true,
            "source_execution":"not_inferred_from_request_construction",
            "actual_input_tokens":"recorded_from_engine_frontier_and_checked_across_phases",
        });
        Ok((request, evidence))
    }

    fn build_request(
        &self,
        manifest: &manifest::Manifest,
        index: usize,
        model_id: &ModelId,
        output_protocol: ferrum_types::ModelOutputProtocol,
        reference_output: Option<(usize, bool)>,
    ) -> Result<(InferenceRequest, serde_json::Value)> {
        match self {
            Self::Rendered => {
                let prompt = &manifest.prompts[index];
                let mut request =
                    InferenceRequest::new(prompt.rendered_prompt.clone(), model_id.clone());
                request.stream = true;
                request.sampling_params = prompt.sampling.clone();
                request.sampling_params.model_output_protocol = output_protocol;
                if let Some((maximum, ignore_eos)) = reference_output {
                    request.sampling_params.max_tokens = maximum;
                    request
                        .metadata
                        .insert("ferrum_ignore_eos".into(), ignore_eos.into());
                }
                Ok((
                    request,
                    serde_json::json!({"source_id":prompt.source_id,"rendered_prompt_sha256":prompt.rendered_prompt_sha256}),
                ))
            }
            Self::ShareGpt {
                recovered,
                template,
                policy,
            } => {
                let prompt = &recovered.prompts[index];
                let (maximum, _) = reference_output.unwrap_or((
                    prompt.sample.requested_output_tokens as usize,
                    policy.ignore_eos,
                ));
                let body = Self::chat_body(prompt, policy, reference_output);
                let typed = serde_json::from_value(body.clone()).map_err(|error| {
                    FerrumError::config(format!("decode shared Chat request: {error}"))
                })?;
                let request = ferrum_server::axum_server::prepare_model_chat_request(
                    &typed,
                    &model_id.0,
                    template,
                    policy.server_default_enable_thinking,
                    policy.interleaved_system_coalescing,
                )?;
                if request.prompt.len() > 1024 * 1024
                    || request.sampling_params.max_tokens != maximum
                {
                    return Err(FerrumError::config(
                        "rendered ShareGPT input exceeds bound or output limit changed",
                    ));
                }
                let body_bytes = serde_json::to_vec(&body).map_err(|error| {
                    FerrumError::config(format!("hash shared Chat request: {error}"))
                })?;
                let evidence = serde_json::json!({
                    "kind":"frozen_sharegpt_chat", "sample":prompt.sample,
                    "requested_model_name":policy.requested_model_name,
                    "resolved_engine_model_id":model_id,
                    "chat_body_sha256":sharegpt::hex(&Sha256::digest(body_bytes)),
                    "rendered_prompt_sha256":sharegpt::hex(&Sha256::digest(request.prompt.as_bytes())),
                    "actual_input_tokens":"recorded_from_engine_frontier",
                });
                Ok((request, evidence))
            }
        }
    }
}
