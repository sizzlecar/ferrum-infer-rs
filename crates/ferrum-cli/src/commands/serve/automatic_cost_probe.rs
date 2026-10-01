//! Resolve probe requests through the same public pure preparation functions
//! as the server. No engine-side replacement chat template is introduced.
use crate::commands::automatic_cost_probe::PromptModel;
use ferrum_engine::{
    AutomaticCostProbeOutput, AutomaticCostProbePromptRenderer, AutomaticCostProbeTemplate,
};
use ferrum_server::{
    axum_server::{prepare_model_chat_request, prepare_model_completion_request},
    chat_template::ModelChatTemplate,
    openai::{ChatMessage, MessageRole},
    ChatCompletionsRequest, CompletionPrompt, CompletionsRequest, StreamOptions,
};
use ferrum_types::{EngineConfig, FerrumError, ResponseFormat, Result};
use std::sync::Arc;

const USER_TEXT: &str = "Explain how to count a sequence of objects, with several examples.";

pub(super) fn templates(
    config: &EngineConfig,
    model_template: Option<&ModelChatTemplate>,
    response_model: &str,
    default_enable_thinking: Option<bool>,
    interleaved_system_coalescing: bool,
) -> Vec<AutomaticCostProbeTemplate> {
    let ferrum_types::SloLiveStructuredCalibration::AutomaticV1 { settings } = &config
        .scheduler
        .slo
        .cost_observation
        .live_structured_calibration
    else {
        return Vec::new();
    };
    if config.scheduler.slo.mode == ferrum_types::SloMode::Off {
        return Vec::new();
    }
    let maximum = settings.cost_probe.maximum_output_tokens.get();
    let mut templates = Vec::new();
    if let Some(model_template) = model_template {
        // Keep the actual configured request and, when supported by the real
        // renderer, also sample the public enable_thinking=false request. Its
        // completion boundary and token masks are set only by the converter.
        let disabled_variant = model_template.reasoning_enabled(default_enable_thinking)
            && model_template.supports_thinking_control();
        for thinking_override in [None, Some(false)]
            .into_iter()
            .take(1 + usize::from(disabled_variant))
        {
            for usage in [false, true] {
                match chat_template(
                    config,
                    model_template,
                    response_model,
                    maximum,
                    usage,
                    default_enable_thinking,
                    interleaved_system_coalescing,
                    thinking_override,
                ) {
                    Ok(template) => templates.push(template),
                    Err(error) => {
                        tracing::warn!(%error,include_usage=usage,?thinking_override,"automatic cost probe Chat template unavailable")
                    }
                }
            }
        }
    } else {
        tracing::warn!(
            "automatic cost probe Chat template unavailable: actual model template missing"
        );
    }
    match completion_template(config, response_model, maximum) {
        Ok(template) => templates.push(template),
        Err(error) => {
            tracing::warn!(%error,"automatic cost probe Completions template unavailable")
        }
    }
    templates
}

#[allow(clippy::too_many_arguments)]
fn chat_template(
    config: &EngineConfig,
    model_template: &ModelChatTemplate,
    response_model: &str,
    maximum: usize,
    include_usage: bool,
    default_enable_thinking: Option<bool>,
    interleaved_system_coalescing: bool,
    thinking_override: Option<bool>,
) -> Result<AutomaticCostProbeTemplate> {
    let s = &config.sampling.default_params;
    // These parameters are not exposed by the ordinary Chat endpoint. Do not
    // silently replace them while claiming the configured policy was sampled.
    if s.tfs.is_some()
        || s.typical_p.is_some()
        || s.mirostat.is_some()
        || s.response_format != ResponseFormat::Text
    {
        return Err(FerrumError::unsupported(
            "configured sampler cannot be represented by the plain Chat probe endpoint",
        ));
    }
    let request =
        ChatCompletionsRequest {
            model: response_model.into(),
            messages: vec![ChatMessage {
                role: MessageRole::User,
                content: USER_TEXT.into(),
                reasoning: None,
                name: None,
                tool_calls: None,
                tool_call_id: None,
                function_call: None,
            }],
            max_tokens: Some(u32::try_from(maximum).map_err(|_| {
                FerrumError::invalid_request("probe output exceeds API token range")
            })?),
            max_completion_tokens: None,
            temperature: Some(s.temperature),
            top_p: Some(s.top_p),
            top_k: s
                .top_k
                .map(i64::try_from)
                .transpose()
                .map_err(|_| FerrumError::invalid_request("configured top_k exceeds API range"))?,
            min_p: s.min_p,
            repetition_penalty: Some(s.repetition_penalty),
            n: Some(1),
            stream: Some(true),
            ignore_eos: None,
            stop: Some(s.stop_sequences.clone()),
            presence_penalty: Some(s.presence_penalty),
            frequency_penalty: Some(s.frequency_penalty),
            logit_bias: None,
            logprobs: None,
            top_logprobs: None,
            user: None,
            seed: s.seed,
            response_format: None,
            reasoning_effort: None,
            tools: None,
            tool_choice: None,
            stream_options: Some(StreamOptions {
                include_usage: Some(include_usage),
            }),
            functions: None,
            function_call: None,
            metadata: None,
            chat_template_kwargs: thinking_override.map(|enabled| {
                [("enable_thinking".to_owned(), serde_json::json!(enabled))]
                    .into_iter()
                    .collect()
            }),
        };
    let prepared = prepare_model_chat_request(
        &request,
        config.model.model_id.as_str(),
        model_template,
        default_enable_thinking,
        interleaved_system_coalescing,
    )?;
    let renderer = ChatRenderer {
        request: serde_json::to_vec(&request)
            .map_err(|e| FerrumError::internal(format!("probe Chat request: {e}")))?
            .into(),
        model: PromptModel::new(model_template)?,
        model_id: config.model.model_id.as_str().into(),
        response_model: response_model.into(),
        default_enable_thinking,
        interleaved_system_coalescing,
        include_usage,
    };
    Ok(AutomaticCostProbeTemplate::new(
        prepared,
        AutomaticCostProbeOutput::ApiChatSse { include_usage },
    )?
    .with_response_model(response_model.into())
    .with_prompt_renderer(Arc::new(renderer)))
}

fn completion_template(
    config: &EngineConfig,
    response_model: &str,
    maximum: usize,
) -> Result<AutomaticCostProbeTemplate> {
    // Actual endpoint conversion owns defaults: notably repetition=1, and
    // no Chat template/ignore-EOS/user-controlled repetition extension.
    let request =
        CompletionsRequest {
            model: response_model.into(),
            prompt: CompletionPrompt::Text(USER_TEXT.into()),
            max_tokens: Some(u32::try_from(maximum).map_err(|_| {
                FerrumError::invalid_request("probe output exceeds API token range")
            })?),
            temperature: None,
            top_p: None,
            n: Some(1),
            stream: Some(true),
            stop: None,
            logprobs: None,
            logit_bias: None,
        };
    let prepared = prepare_model_completion_request(&request, config.model.model_id.as_str())?;
    let renderer = CompletionRenderer {
        request: serde_json::to_vec(&request)
            .map_err(|e| FerrumError::internal(format!("probe Completion request: {e}")))?
            .into(),
        model_id: config.model.model_id.as_str().into(),
        response_model: response_model.into(),
    };
    Ok(
        AutomaticCostProbeTemplate::new(prepared, AutomaticCostProbeOutput::ApiCompletionSse)?
            .with_response_model(response_model.into())
            .with_prompt_renderer(Arc::new(renderer)),
    )
}

#[derive(Debug)]
struct ChatRenderer {
    request: Arc<[u8]>,
    model: PromptModel,
    model_id: Arc<str>,
    response_model: Arc<str>,
    default_enable_thinking: Option<bool>,
    interleaved_system_coalescing: bool,
    include_usage: bool,
}
impl AutomaticCostProbePromptRenderer for ChatRenderer {
    fn render_user_text(&self, text: &str) -> Result<AutomaticCostProbeTemplate> {
        let mut request: ChatCompletionsRequest = serde_json::from_slice(&self.request)
            .map_err(|e| FerrumError::internal(format!("probe Chat renderer: {e}")))?;
        request.messages[0].content = text.into();
        let prepared = prepare_model_chat_request(
            &request,
            &self.model_id,
            &self.model.resolve()?,
            self.default_enable_thinking,
            self.interleaved_system_coalescing,
        )?;
        Ok(AutomaticCostProbeTemplate::new(
            prepared,
            AutomaticCostProbeOutput::ApiChatSse {
                include_usage: self.include_usage,
            },
        )?
        .with_response_model(self.response_model.to_string()))
    }
    fn retained_payload_bytes(&self) -> Option<usize> {
        std::mem::size_of::<Self>()
            .checked_add(self.request.len())?
            .checked_add(self.model.retained_payload_bytes()?)?
            .checked_add(self.model_id.len())?
            .checked_add(self.response_model.len())?
            .checked_add(6 * std::mem::size_of::<usize>())
    }
}

#[derive(Debug)]
struct CompletionRenderer {
    request: Arc<[u8]>,
    model_id: Arc<str>,
    response_model: Arc<str>,
}
impl AutomaticCostProbePromptRenderer for CompletionRenderer {
    fn render_user_text(&self, text: &str) -> Result<AutomaticCostProbeTemplate> {
        let mut request: CompletionsRequest = serde_json::from_slice(&self.request)
            .map_err(|e| FerrumError::internal(format!("probe Completion renderer: {e}")))?;
        request.prompt = CompletionPrompt::Text(text.into());
        let prepared = prepare_model_completion_request(&request, &self.model_id)?;
        Ok(
            AutomaticCostProbeTemplate::new(prepared, AutomaticCostProbeOutput::ApiCompletionSse)?
                .with_response_model(self.response_model.to_string()),
        )
    }
    fn retained_payload_bytes(&self) -> Option<usize> {
        std::mem::size_of::<Self>()
            .checked_add(self.request.len())?
            .checked_add(self.model_id.len())?
            .checked_add(self.response_model.len())?
            .checked_add(6 * std::mem::size_of::<usize>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_types::{ApiRequest, SloAutomaticCostProbeSamplingPresetV1};

    mod benchmark_policy;

    #[test]
    fn automatic_cost_probe_serve_long_context_uses_chat_and_completion_converters() {
        let config = EngineConfig::default();
        let model = ModelChatTemplate::new(
            "{% for m in messages %}{{ m['role'] }}={{ m['content'] }}\n{% endfor %}assistant=",
            "renderer fixture",
        );
        for usage in [false, true] {
            let original =
                chat_template(&config, &model, "public-name", 16, usage, None, true, None).unwrap();
            let variant = crate::commands::automatic_cost_probe::tests::variant(&original);
            let request = variant.template.resolved_request().unwrap();
            assert!(request.prompt.starts_with("user="));
            assert!(request.prompt.ends_with("assistant="));
            let Some(ApiRequest::Chat(chat)) = request.api_request else {
                panic!("original Chat API")
            };
            assert!(chat.messages[0].content.contains("object"));
            assert_eq!(chat.stream_options.unwrap().include_usage, Some(usage));
            assert_eq!(variant.template.response_model(), "public-name");
            let mut expected_metadata = original.resolved_request().unwrap().metadata;
            expected_metadata.get_mut("openai_messages").unwrap()[0]["content"] =
                serde_json::json!(chat.messages[0].content);
            assert_eq!(request.metadata, expected_metadata);
        }
        let original = completion_template(&config, "public-name", 16).unwrap();
        let variant = crate::commands::automatic_cost_probe::tests::variant(&original);
        let request = variant.template.resolved_request().unwrap();
        let Some(ApiRequest::Completion(api)) = request.api_request else {
            panic!("original Completion API")
        };
        assert_eq!(request.prompt, api.prompt);
        assert_eq!(request.sampling_params.repetition_penalty, 1.0);
        assert_eq!(variant.template.response_model(), "public-name");
        assert_eq!(
            variant.template.output(),
            AutomaticCostProbeOutput::ApiCompletionSse
        );
    }
    #[test]
    fn automatic_cost_probe_serve_uses_product_template_api_and_endpoint_policies() {
        let mut config = EngineConfig::default();
        config.sampling.default_params.repetition_penalty = 1.25;
        config.sampling.default_params.stop_sequences = vec!["user-stop".into()];
        let model = ModelChatTemplate::new(
            "{% for m in messages %}{{ m['role'] }}={{ m['content'] }}\n{% endfor %}assistant=",
            "typed fixture",
        );
        for usage in [false, true] {
            let t = chat_template(&config, &model, "public-alias", 16, usage, None, true, None)
                .unwrap();
            let r = t.resolved_request().unwrap();
            assert_eq!(r.model_id, config.model.model_id);
            assert!(r.prompt.starts_with("user="));
            assert!(r.prompt.ends_with("assistant="));
            assert_eq!(r.sampling_params.repetition_penalty, 1.25);
            assert_eq!(r.sampling_params.stop_sequences, ["user-stop"]);
            let Some(ApiRequest::Chat(chat)) = &r.api_request else {
                panic!("actual Chat request");
            };
            assert_eq!(chat.messages[0].content, USER_TEXT);
            assert_eq!(
                chat.stream_options.as_ref().unwrap().include_usage,
                Some(usage)
            );
            assert_eq!(
                r.metadata[ferrum_types::PROMPT_OPENED_REASONING_METADATA_KEY],
                false
            );
        }
        let completion = completion_template(&config, "public-alias", 16).unwrap();
        assert!(matches!(
            completion.resolved_request().unwrap().api_request,
            Some(ApiRequest::Completion(_))
        ));
        assert_eq!(
            completion
                .resolved_request()
                .unwrap()
                .sampling_params
                .repetition_penalty,
            1.0
        );
        assert!(completion
            .resolved_request()
            .unwrap()
            .sampling_params
            .stop_sequences
            .is_empty());
        assert!(!completion.supports_preset(SloAutomaticCostProbeSamplingPresetV1::GreedyLength));
    }
    fn automatic_config() -> EngineConfig {
        let mut config = EngineConfig::default();
        config.scheduler.slo.mode = ferrum_types::SloMode::Observe;
        config
            .scheduler
            .slo
            .cost_observation
            .live_structured_calibration =
            ferrum_types::SloLiveStructuredCalibration::AutomaticV1 {
                settings: Default::default(),
            };
        config.sampling.default_params = ferrum_server::default_chat_sampling_params();
        config
    }

    #[test]
    fn automatic_cost_probe_serve_thinking_variant_uses_real_public_renderer() {
        use ferrum_types::{
            ModelOutputProtocol, ResponseCompletionBoundary, StructuredOutputStart,
        };
        let config = automatic_config();
        let model = ModelChatTemplate::new(
            include_str!("../../../../ferrum-server/tests/fixtures/chat_template/Qwen__Qwen3.5-35B-A3B/template.jinja"),
            "arbitrary capability fixture",
        );
        assert!(model.reasoning_enabled(None));
        assert!(model.supports_thinking_control());
        let generated = templates(&config, Some(&model), "public-alias", None, true);
        let requests: Vec<_> = generated
            .iter()
            .filter(|t| matches!(t.output(), AutomaticCostProbeOutput::ApiChatSse { .. }))
            .map(|t| (t.output(), t.resolved_request().unwrap()))
            .collect();
        for usage in [false, true] {
            let matching: Vec<_> = requests
                .iter()
                .filter(|(out, _)| {
                    *out == AutomaticCostProbeOutput::ApiChatSse {
                        include_usage: usage,
                    }
                })
                .collect();
            assert_eq!(
                matching.len(),
                2,
                "configured and real public disabled request"
            );
            let configured = &matching[0].1;
            let disabled = &matching[1].1;
            for request in [configured, disabled] {
                assert_eq!(
                    request.sampling_params.model_output_protocol,
                    ModelOutputProtocol::Text
                );
                assert_eq!(
                    request.sampling_params.structured_output_start,
                    StructuredOutputStart::Immediate
                );
                assert_eq!(
                    request.sampling_params.repetition_penalty,
                    config.sampling.default_params.repetition_penalty
                );
                assert_eq!(
                    request.sampling_params.temperature,
                    config.sampling.default_params.temperature
                );
                assert_eq!(
                    request.sampling_params.stop_sequences,
                    config.sampling.default_params.stop_sequences
                );
                assert!(matches!(&request.api_request, Some(ApiRequest::Chat(_))));
                assert!(!request.metadata.contains_key("ferrum_ignore_eos"));
            }
            assert!(matches!(
                configured.sampling_params.response_completion_boundary,
                ResponseCompletionBoundary::AfterDelimiterAndPayload { .. }
            ));
            assert_eq!(
                disabled.sampling_params.response_completion_boundary,
                ResponseCompletionBoundary::Immediate
            );
            assert_eq!(
                configured.metadata[ferrum_types::PROMPT_OPENED_REASONING_METADATA_KEY],
                true
            );
            assert_eq!(
                disabled.metadata[ferrum_types::PROMPT_OPENED_REASONING_METADATA_KEY],
                false
            );
            assert!(disabled.metadata["ferrum_initial_forbidden_token_texts"]
                .as_array()
                .unwrap()
                .iter()
                .any(|value| value == "<think>"));
            assert_ne!(configured.prompt, disabled.prompt);
        }
        // The public service default remains untouched after probe construction.
        let original = chat_template(&config, &model, "public-alias", 16, false, None, true, None)
            .unwrap()
            .resolved_request()
            .unwrap();
        assert!(matches!(
            original.sampling_params.response_completion_boundary,
            ResponseCompletionBoundary::AfterDelimiterAndPayload { .. }
        ));
        let already_disabled = templates(&config, Some(&model), "public-alias", Some(false), true);
        assert_eq!(
            already_disabled
                .iter()
                .filter(|t| matches!(t.output(), AutomaticCostProbeOutput::ApiChatSse { .. }))
                .count(),
            2
        );
    }

    #[test]
    fn automatic_cost_probe_serve_no_fabricated_thinking_switch_and_off_is_unchanged() {
        let mut config = automatic_config();
        let model = ModelChatTemplate::new(
            "{% for m in messages %}{{ m['content'] }}{% endfor %}{% if add_generation_prompt %}<think>\n{% endif %}",
            "always opened",
        );
        assert!(model.reasoning_enabled(None));
        assert!(!model.supports_thinking_control());
        let generated = templates(&config, Some(&model), "public-alias", None, true);
        let chats: Vec<_> = generated
            .iter()
            .filter(|t| matches!(t.output(), AutomaticCostProbeOutput::ApiChatSse { .. }))
            .collect();
        assert_eq!(chats.len(), 2);
        assert!(chats.iter().all(|t| matches!(
            t.resolved_request()
                .unwrap()
                .sampling_params
                .response_completion_boundary,
            ferrum_types::ResponseCompletionBoundary::AfterDelimiterAndPayload { .. }
        )));
        config.scheduler.slo.mode = ferrum_types::SloMode::Off;
        assert!(templates(&config, Some(&model), "public-alias", None, true).is_empty());
    }
}
