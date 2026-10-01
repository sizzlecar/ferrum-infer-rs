//! Resolve private CLI probes through the ordinary run renderer and policy.
use super::*;
use crate::commands::automatic_cost_probe::PromptModel;
use ferrum_engine::{
    AutomaticCostProbeOutput, AutomaticCostProbePromptRenderer, AutomaticCostProbeTemplate,
};
use std::sync::Arc;

const USER_TEXT: &str = "Explain how to count a sequence of objects, with several examples.";

pub(super) fn templates(
    config: &ferrum_types::EngineConfig,
    command: &RunCommand,
    model_template: Option<&ModelChatTemplate>,
    options: &ChatTemplateOptions,
) -> Vec<AutomaticCostProbeTemplate> {
    if config.scheduler.slo.mode == ferrum_types::SloMode::Off
        || !matches!(
            config
                .scheduler
                .slo
                .cost_observation
                .live_structured_calibration,
            ferrum_types::SloLiveStructuredCalibration::AutomaticV1 { .. }
        )
    {
        return Vec::new();
    }
    resolved_templates(config, command.system.as_deref(), model_template, options)
}

fn resolved_templates(
    config: &ferrum_types::EngineConfig,
    system: Option<&str>,
    model_template: Option<&ModelChatTemplate>,
    options: &ChatTemplateOptions,
) -> Vec<AutomaticCostProbeTemplate> {
    let mut templates = Vec::new();
    match resolved_template(config, system, model_template, options) {
        Ok(template) => templates.push(template),
        Err(error) => tracing::warn!(%error,"automatic cost probe CLI template unavailable"),
    }
    // This is a second actual public CLI policy, never a replacement of the
    // caller's default. A rendered capability must prove the switch exists.
    if options.reasoning_effort.is_none()
        && model_template.is_some_and(|model| {
            model.reasoning_enabled(options.enable_thinking) && model.supports_thinking_control()
        })
    {
        let mut disabled = options.clone();
        disabled.enable_thinking = Some(false);
        match resolved_template(config, system, model_template, &disabled) {
            Ok(template) => templates.push(template),
            Err(error) => {
                tracing::warn!(%error,"automatic cost probe CLI reasoning-disabled template unavailable")
            }
        }
    }
    templates
}

fn resolved_template(
    config: &ferrum_types::EngineConfig,
    system: Option<&str>,
    model_template: Option<&ModelChatTemplate>,
    options: &ChatTemplateOptions,
) -> Result<AutomaticCostProbeTemplate> {
    let prompt = build_chat_prompt(
        &[],
        USER_TEXT,
        system,
        config.model.model_id.as_str(),
        model_template,
        options,
    )?;
    let mut sampling = config.sampling.default_params.clone();
    sampling.model_output_protocol = model_template
        .map(|m| m.output_protocol)
        .unwrap_or(ferrum_types::ModelOutputProtocol::Text);
    let sampling = sampling_params_for_prompt(sampling, &prompt);
    let metadata = run_request_metadata(
        &prompt,
        options,
        sampling.model_output_protocol,
        model_template,
    );
    let mut request = InferenceRequest::new(prompt, config.model.model_id.clone());
    request.stream = true; // same credited CLI handoff as execute_observed
    request.sampling_params = sampling;
    request.metadata = metadata;
    let base = AutomaticCostProbeTemplate::new(request, AutomaticCostProbeOutput::CliText)?;
    let renderer = RunRenderer {
        base: base.clone(),
        system: system.map(Arc::<str>::from),
        model: model_template.map(PromptModel::new).transpose()?,
        options: options.clone(),
    };
    Ok(base.with_prompt_renderer(Arc::new(renderer)))
}

#[derive(Debug)]
struct RunRenderer {
    base: AutomaticCostProbeTemplate,
    system: Option<Arc<str>>,
    model: Option<PromptModel>,
    options: ChatTemplateOptions,
}
impl AutomaticCostProbePromptRenderer for RunRenderer {
    fn render_user_text(&self, text: &str) -> Result<AutomaticCostProbeTemplate> {
        let model = self.model.as_ref().map(PromptModel::resolve).transpose()?;
        let mut request = self.base.resolved_request()?;
        request.prompt = build_chat_prompt(
            &[],
            text,
            self.system.as_deref(),
            request.model_id.as_str(),
            model.as_ref(),
            &self.options,
        )?;
        request.sampling_params =
            sampling_params_for_prompt(request.sampling_params, &request.prompt);
        request.metadata = run_request_metadata(
            &request.prompt,
            &self.options,
            request.sampling_params.model_output_protocol,
            model.as_ref(),
        );
        AutomaticCostProbeTemplate::new(request, AutomaticCostProbeOutput::CliText)
    }
    fn retained_payload_bytes(&self) -> Option<usize> {
        std::mem::size_of::<Self>()
            .checked_add(self.base.retained_payload_bytes()?)?
            .checked_add(
                self.system
                    .as_ref()
                    .map_or(0, |s| s.len() + 2 * std::mem::size_of::<usize>()),
            )?
            .checked_add(
                self.model
                    .as_ref()
                    .map_or(Some(0), PromptModel::retained_payload_bytes)?,
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn automatic_cost_probe_run_long_context_keeps_real_renderer_system_and_policy() {
        let mut config = ferrum_types::EngineConfig::default();
        config.sampling.default_params.stop_sequences = vec!["actual-stop".into()];
        let model = ModelChatTemplate::new(
            "{% for m in messages %}{{ m['role'] }}={{ m['content'] }}\n{% endfor %}assistant=",
            "renderer fixture",
        );
        let original = resolved_template(
            &config,
            Some("keep system"),
            Some(&model),
            &ChatTemplateOptions::default(),
        )
        .unwrap();
        let before = original.serialized_request().to_vec();
        let variant = crate::commands::automatic_cost_probe::tests::variant(&original);
        let request = variant.template.resolved_request().unwrap();
        assert!(request.prompt.contains("system=keep system"));
        assert!(request.prompt.contains("user="));
        assert!(request.prompt.ends_with("assistant="));
        assert_eq!(request.sampling_params.stop_sequences, ["actual-stop"]);
        assert_eq!(
            request.metadata,
            original.resolved_request().unwrap().metadata
        );
        assert_eq!(variant.template.output(), AutomaticCostProbeOutput::CliText);
        assert!(request.api_request.is_none());
        assert_eq!(original.serialized_request(), before);
    }
    #[test]
    fn automatic_cost_probe_run_uses_actual_template_and_configured_sampling() {
        let mut config = ferrum_types::EngineConfig::default();
        config.sampling.default_params.repetition_penalty = 1.25;
        config.sampling.default_params.stop_sequences = vec!["user-stop".into()];
        let template = ModelChatTemplate::new(
            "{% for m in messages %}{{ m['role'] }}={{ m['content'] }}\n{% endfor %}assistant=",
            "typed fixture",
        );
        let probe = resolved_template(
            &config,
            Some("a system instruction"),
            Some(&template),
            &ChatTemplateOptions::default(),
        )
        .unwrap();
        let request = probe.resolved_request().unwrap();
        assert!(request.prompt.contains("system=a system instruction"));
        assert!(request.prompt.contains(USER_TEXT));
        assert!(request.prompt.ends_with("assistant="));
        assert_eq!(request.sampling_params.repetition_penalty, 1.25);
        assert_eq!(request.sampling_params.stop_sequences, ["user-stop"]);
        assert!(request.api_request.is_none());
        assert_eq!(probe.output(), AutomaticCostProbeOutput::CliText);
    }
    #[test]
    fn automatic_cost_probe_run_thinking_variant_keeps_normal_policy() {
        let mut config = ferrum_types::EngineConfig::default();
        config.sampling.default_params.repetition_penalty = 1.25;
        config.sampling.default_params.stop_sequences = vec!["user-stop".into()];
        let model = ModelChatTemplate::new(
            include_str!("../../../../ferrum-server/tests/fixtures/chat_template/Qwen__Qwen3.5-35B-A3B/template.jinja"),
            "arbitrary capability fixture",
        );
        let options = ChatTemplateOptions::default();
        let generated = resolved_templates(&config, Some("keep system"), Some(&model), &options);
        assert_eq!(generated.len(), 2);
        let configured = generated[0].resolved_request().unwrap();
        let disabled = generated[1].resolved_request().unwrap();
        assert!(matches!(
            configured.sampling_params.response_completion_boundary,
            ferrum_types::ResponseCompletionBoundary::AfterDelimiterAndPayload { .. }
        ));
        assert_eq!(
            disabled.sampling_params.response_completion_boundary,
            ferrum_types::ResponseCompletionBoundary::Immediate
        );
        for request in [&configured, &disabled] {
            assert_eq!(
                request.sampling_params.model_output_protocol,
                ferrum_types::ModelOutputProtocol::Text
            );
            assert_eq!(request.sampling_params.repetition_penalty, 1.25);
            assert_eq!(request.sampling_params.stop_sequences, ["user-stop"]);
            assert!(request.prompt.contains("keep system"));
            assert!(request.api_request.is_none());
        }
        assert_eq!(options.enable_thinking, None);
        assert!(
            disabled.metadata[RUN_INITIAL_FORBIDDEN_TOKEN_TEXTS_METADATA_KEY]
                .as_array()
                .unwrap()
                .iter()
                .any(|value| value == "<think>")
        );
        let already_disabled = ChatTemplateOptions {
            enable_thinking: Some(false),
            ..Default::default()
        };
        assert_eq!(
            resolved_templates(&config, None, Some(&model), &already_disabled).len(),
            1
        );
        let always_open = ModelChatTemplate::new(
            "{% for m in messages %}{{ m['content'] }}{% endfor %}{% if add_generation_prompt %}<think>\n{% endif %}",
            "always opened",
        );
        assert!(!always_open.supports_thinking_control());
        assert_eq!(
            resolved_templates(&config, None, Some(&always_open), &options).len(),
            1
        );
    }
}
