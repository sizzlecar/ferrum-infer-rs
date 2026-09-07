//! Standard reasoning controls must reach the selected template through HTTP.
use super::*;
use ferrum_engine::ContinuousBatchEngine;
use ferrum_interfaces::{sampler::GreedySampler, Tokenizer};
use ferrum_scheduler::ContinuousBatchScheduler;
use ferrum_testkit::MockTensorFactory;
use ferrum_tokenizer::HuggingFaceTokenizer;
use ferrum_types::{FerrumError, ReasoningEffortSupport};
use futures::FutureExt;
use std::{
    collections::BTreeSet,
    panic::{resume_unwind, AssertUnwindSafe},
    sync::atomic::Ordering,
    time::Duration,
};
use tokenizers::{
    decoders::fuse::Fuse,
    models::bpe::{Vocab, BPE},
    AddedToken,
};

// Reuse the existing executor contract fixture: only logits/cache storage are
// simulated; the production engine tokenizes, prefills, samples, and completes.
#[path = "engine_stop_contract/executor.rs"]
mod executor;
use executor::{LogitStep, ScriptedExecutor};

const EOS: &str = "<|endoftext|>";
const EFFORT_TEMPLATE: &str =
    "Reasoning: {{ reasoning_effort | default('template-default') }}\n{{ messages[-1].content }}";

#[derive(Clone, Copy, Debug)]
enum Endpoint {
    Chat,
    Responses,
}

impl Endpoint {
    fn path(self) -> &'static str {
        match self {
            Self::Chat => "/v1/chat/completions",
            Self::Responses => "/v1/responses",
        }
    }

    fn request(self, effort: Option<Value>) -> Value {
        let mut wire = match self {
            Self::Chat => json!({
                "model": "served-alias",
                "messages": [{"role": "user", "content": "hello"}],
            }),
            Self::Responses => json!({
                "model": "served-alias",
                "input": "hello",
            }),
        };
        if let Some(effort) = effort {
            match self {
                Self::Chat => wire["reasoning_effort"] = effort,
                Self::Responses => wire["reasoning"] = json!({"effort": effort}),
            }
        }
        wire
    }
}

async fn assert_success(response: Response, endpoint: Endpoint, stream: bool) {
    assert_eq!(response.status(), AxumStatusCode::OK, "{endpoint:?}");
    if stream {
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            "text/event-stream",
        );
        let body = response_text(response).await;
        let events = responses_sse_json_events(&body);
        assert!(!events.is_empty(), "{endpoint:?}: {body}");
        assert!(
            events.iter().all(|event| event.get("error").is_none()),
            "{endpoint:?}: {body}",
        );
        match endpoint {
            Endpoint::Chat => {
                assert!(body.contains("data: [DONE]"), "{body}");
                assert!(
                    events
                        .iter()
                        .any(|event| { event["choices"][0]["delta"]["content"] == "captured" }),
                    "{body}"
                );
            }
            Endpoint::Responses => assert!(
                events.iter().any(|event| {
                    event["type"] == "response.completed"
                        && event["response"]["status"] == "completed"
                }),
                "{body}",
            ),
        }
    } else {
        let body = response_json(response).await;
        match endpoint {
            Endpoint::Chat => assert_eq!(body["choices"][0]["message"]["content"], "captured"),
            Endpoint::Responses => assert_eq!(body["status"], "completed"),
        }
    }
}

async fn assert_invalid_request(response: Response, engine: &CapturingLlm) {
    assert_eq!(response.status(), AxumStatusCode::BAD_REQUEST);
    let body = response_json(response).await;
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("reasoning"),
        "the error must identify the rejected control: {body}",
    );
    assert!(
        !engine.has_captured_request(),
        "invalid controls must not enter inference",
    );
}

struct EngineObservation {
    body: Value,
    decoded_inputs: String,
    prefill: String,
}

async fn infer_with_real_engine(
    template: ModelChatTemplate,
    mut wire: Value,
    steps: &[&[(&str, f32)]],
) -> EngineObservation {
    let mut vocab = Vocab::new();
    for piece in (32u8..=126).map(|byte| (byte as char).to_string()).chain([
        "\n".to_owned(),
        EOS.to_owned(),
        "<think>".to_owned(),
    ]) {
        let id = vocab.len() as u32;
        vocab.insert(piece, id);
    }
    let mut inner = tokenizers::Tokenizer::new(
        BPE::builder()
            .vocab_and_merges(vocab, vec![])
            .build()
            .unwrap(),
    );
    inner.with_decoder(Some(Fuse::new()));
    inner.add_special_tokens(&[
        AddedToken::from(EOS, true),
        AddedToken::from("<think>", true),
    ]);
    let generation = json!({"eos_token_id": inner.token_to_id(EOS).unwrap()});
    let tokenizer = Arc::new(
        HuggingFaceTokenizer::from_source_bytes(
            inner.to_string(false).unwrap().as_bytes(),
            None,
            Some(generation.to_string().as_bytes()),
        )
        .await
        .unwrap(),
    );
    let executor = Arc::new(if steps.iter().all(|step| step.len() == 1) {
        ScriptedExecutor::new(
            tokenizer.vocab_size(),
            steps
                .iter()
                .map(|step| tokenizer.token_id(step[0].0).unwrap())
                .collect(),
        )
    } else {
        ScriptedExecutor::from_steps(
            tokenizer.vocab_size(),
            steps
                .iter()
                .map(|step| {
                    LogitStep::candidates(
                        step.iter()
                            .map(|(token, logit)| (tokenizer.token_id(token).unwrap(), *logit))
                            .collect(),
                    )
                })
                .collect(),
        )
    });
    let mut config = EngineConfig::default();
    config.model.model_id = ModelId::new("reasoning-contract");
    config.scheduler.max_running_requests = 1;
    config.batching.max_num_batched_tokens = 256;
    let engine = Arc::new(
        ContinuousBatchEngine::new_plan_runtime(
            config.clone(),
            Arc::new(ContinuousBatchScheduler::new(config.scheduler)),
            tokenizer.clone(),
            Arc::new(GreedySampler),
            executor.clone(),
            Arc::new(MockTensorFactory),
        )
        .unwrap(),
    );
    let router = AxumServer::from_llm(engine.clone())
        .with_prompt_template(Some(template))
        .build_router();
    wire["model"] = json!("reasoning-contract");
    wire["temperature"] = json!(0);
    wire["max_tokens"] = json!(8);
    let outcome = AssertUnwindSafe(tokio::time::timeout(Duration::from_secs(5), async {
        let response = post_json(router, Endpoint::Chat.path(), wire).await;
        (response.status(), response_text(response).await)
    }))
    .catch_unwind()
    .await;
    let shutdown = tokio::time::timeout(Duration::from_secs(5), engine.shutdown()).await;
    executor.assert_released();
    shutdown.expect("engine shutdown must terminate").unwrap();
    let (status, body) = match outcome {
        Ok(response) => response.expect("engine HTTP request must terminate"),
        Err(panic) => resume_unwind(panic),
    };
    assert_eq!(status, AxumStatusCode::OK, "{body}");
    let body: Value = serde_json::from_str(&body).unwrap();
    executor.assert_completed();
    EngineObservation {
        body,
        decoded_inputs: tokenizer.decode(&executor.decoded_inputs(), false).unwrap(),
        prefill: tokenizer.decode(&executor.prefill_tokens(), false).unwrap(),
    }
}

#[tokio::test]
async fn standard_chat_low_reaches_real_engine_prefill() {
    let observation = infer_with_real_engine(
        ModelChatTemplate::new(EFFORT_TEMPLATE, "effort-contract"),
        Endpoint::Chat.request(Some(json!("low"))),
        &[&[("A", 1.0)], &[(EOS, 1.0)]],
    )
    .await;
    assert_eq!(observation.body["choices"][0]["message"]["content"], "A");
    assert_eq!(observation.decoded_inputs, "A");
    assert_eq!(
        observation.prefill, "Reasoning: low\nhello",
        "standard wire effort must survive rendering and actual engine tokenization",
    );
}

#[tokio::test]
async fn none_does_not_add_a_thinking_token_mask_to_unclassified_templates() {
    let requires_system_message = concat!(
        "{% if messages[0].role != 'system' %}",
        "{{ raise_exception('a system message is required') }}{% endif %}",
        "{{ messages[0].content }}|{{ messages[1].content }}",
    );
    for (source, protocol) in [
        (
            "{{ messages[-1].content }}",
            ferrum_types::ModelReasoningProtocol::None,
        ),
        (
            requires_system_message,
            ferrum_types::ModelReasoningProtocol::Unknown,
        ),
    ] {
        let template = ModelChatTemplate::new(source, "unclassified-framing-contract");
        assert_eq!(template.reasoning_protocol, protocol);
        for effort in [None, Some(json!("none"))] {
            let mut wire = Endpoint::Chat.request(effort);
            wire["messages"] = json!([
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "hello"},
            ]);
            let observation = infer_with_real_engine(
                template.clone(),
                wire,
                &[
                    &[("<think>", 2.0), ("A", 1.0)],
                    &[("A", 1.0)],
                    &[(EOS, 1.0)],
                ],
            )
            .await;
            assert_eq!(
                observation.decoded_inputs,
                "<think>A",
                "effort none must not change token sampling without a declared thinking framing protocol",
            );
        }
    }
}

#[tokio::test]
async fn standard_effort_values_reach_sync_and_stream_templates_in_both_apis() {
    for endpoint in [Endpoint::Chat, Endpoint::Responses] {
        for stream in [false, true] {
            let (router, engine) = router_with_capturing_llm_and_template(ModelChatTemplate::new(
                EFFORT_TEMPLATE,
                "effort-only-contract",
            ));
            for effort in ["none", "minimal", "low", "medium", "high", "xhigh", "max"] {
                let mut wire = endpoint.request(Some(json!(effort)));
                wire["stream"] = json!(stream);
                let response = post_json(router.clone(), endpoint.path(), wire).await;
                assert_success(response, endpoint, stream).await;
                assert_eq!(
                    engine.last_request().prompt,
                    format!("Reasoning: {effort}\nhello"),
                    "{endpoint:?}, stream={stream}: preserve the requested effort, including none",
                );
            }
        }
    }
}

#[tokio::test]
async fn malformed_standard_effort_is_rejected_before_inference_in_both_apis() {
    for endpoint in [Endpoint::Chat, Endpoint::Responses] {
        for invalid in [json!(1), json!({"level": "low"}), json!("extreme")] {
            let (router, engine) = router_with_capturing_llm_and_template(ModelChatTemplate::new(
                EFFORT_TEMPLATE,
                "effort-validation-contract",
            ));
            let response =
                post_json(router, endpoint.path(), endpoint.request(Some(invalid))).await;
            assert_invalid_request(response, &engine).await;
        }
    }
}

#[tokio::test]
async fn declared_support_accepts_only_its_efforts_and_preserves_omitted_defaults() {
    for endpoint in [Endpoint::Chat, Endpoint::Responses] {
        for declared in [
            BTreeSet::from([
                ferrum_types::ReasoningEffort::Low,
                ferrum_types::ReasoningEffort::High,
            ]),
            BTreeSet::new(),
        ] {
            let mut template = ModelChatTemplate::new(EFFORT_TEMPLATE, "declared-effort-contract");
            template.reasoning_effort_support = ReasoningEffortSupport::Declared(declared.clone());
            let (router, engine) = router_with_capturing_llm_and_template(template.clone());
            let omitted = post_json(router, endpoint.path(), endpoint.request(None)).await;
            assert_success(omitted, endpoint, false).await;
            assert_eq!(
                engine.last_request().prompt,
                "Reasoning: template-default\nhello"
            );

            // A declaration restricts explicit controls, never the template's own default.
            for effort in ["low", "none", "max"] {
                let (router, engine) = router_with_capturing_llm_and_template(template.clone());
                let response = post_json(
                    router,
                    endpoint.path(),
                    endpoint.request(Some(json!(effort))),
                )
                .await;
                if declared.contains(&effort.parse::<ferrum_types::ReasoningEffort>().unwrap()) {
                    assert_success(response, endpoint, false).await;
                    assert_eq!(
                        engine.last_request().prompt,
                        format!("Reasoning: {effort}\nhello")
                    );
                } else {
                    assert_invalid_request(response, &engine).await;
                }
            }
        }
    }
}

#[tokio::test]
async fn legacy_template_effort_remains_template_owned_with_declared_standard_support() {
    let mut template = ModelChatTemplate::new(EFFORT_TEMPLATE, "declared-extension-contract");
    template.reasoning_effort_support =
        ReasoningEffortSupport::Declared(BTreeSet::from([ferrum_types::ReasoningEffort::Low]));
    let (router, engine) = router_with_capturing_llm_and_template(template);
    let mut wire = Endpoint::Chat.request(None);
    wire["chat_template_kwargs"] = json!({"reasoning_effort": "high"});
    let response = post_json(router, Endpoint::Chat.path(), wire).await;
    assert_success(response, Endpoint::Chat, false).await;
    assert_eq!(engine.last_request().prompt, "Reasoning: high\nhello");
}

#[tokio::test]
async fn legacy_template_effort_preserves_independent_thinking_controls() {
    let template = concat!(
        "Thinking: {% if enable_thinking is not defined %}template-default",
        "{% elif enable_thinking %}enabled<think>{% else %}disabled{% endif %}; ",
        "Reasoning: {{ reasoning_effort }}",
    );
    for (server_default, kwargs, expected_prompt) in [
        (
            None,
            json!({"reasoning_effort": "low"}),
            "Thinking: template-default; Reasoning: low",
        ),
        (
            Some(false),
            json!({"reasoning_effort": "low"}),
            "Thinking: disabled; Reasoning: low",
        ),
        (
            Some(true),
            json!({"enable_thinking": false, "reasoning_effort": "low"}),
            "Thinking: disabled; Reasoning: low",
        ),
    ] {
        let (router, engine) = router_with_capturing_llm_and_template_default(
            ModelChatTemplate::new(template, "legacy-independent-controls-contract"),
            server_default,
        );
        let mut wire = Endpoint::Chat.request(None);
        wire["chat_template_kwargs"] = kwargs;
        let response = post_json(router, Endpoint::Chat.path(), wire).await;
        assert_success(response, Endpoint::Chat, false).await;
        assert_eq!(engine.last_request().prompt, expected_prompt);
    }
}

#[tokio::test]
async fn omitted_and_null_preserve_template_defaults_in_both_apis() {
    let template = concat!(
        "Thinking: {% if enable_thinking is defined %}explicit{% else %}undefined{% endif %}; ",
        "Reasoning: {{ reasoning_effort | default('template-default') }}",
    );
    for endpoint in [Endpoint::Chat, Endpoint::Responses] {
        let (router, engine) = router_with_capturing_llm_and_template(ModelChatTemplate::new(
            template,
            "omitted-controls-contract",
        ));
        let mut requests = vec![endpoint.request(None), endpoint.request(Some(Value::Null))];
        if matches!(endpoint, Endpoint::Responses) {
            let mut null_reasoning = endpoint.request(None);
            null_reasoning["reasoning"] = Value::Null;
            requests.push(null_reasoning);
        }
        for wire in requests {
            let response = post_json(router.clone(), endpoint.path(), wire.clone()).await;
            assert_eq!(response.status(), AxumStatusCode::OK, "{wire}");
            assert_eq!(
                engine.last_request().prompt,
                "Thinking: undefined; Reasoning: template-default",
                "{wire}",
            );
        }
    }
}

#[tokio::test]
async fn positive_effort_overrides_server_disabled_default_in_both_apis() {
    let template = concat!(
        "{% if enable_thinking | default(false) %}",
        "Reasoning: {{ reasoning_effort | default('template-default') }}<think>\n",
        "{% else %}Thinking disabled{% endif %}",
    );
    for endpoint in [Endpoint::Chat, Endpoint::Responses] {
        let (router, engine) = router_with_capturing_llm_and_template_default(
            ModelChatTemplate::new(template, "binary-thinking-contract"),
            Some(false),
        );
        let omitted = post_json(router.clone(), endpoint.path(), endpoint.request(None)).await;
        assert_eq!(omitted.status(), AxumStatusCode::OK, "{endpoint:?}");
        assert_eq!(engine.last_request().prompt, "Thinking disabled");

        let explicit = post_json(
            router,
            endpoint.path(),
            endpoint.request(Some(json!("low"))),
        )
        .await;
        assert_eq!(explicit.status(), AxumStatusCode::OK, "{endpoint:?}");
        assert_eq!(
            engine.last_request().prompt,
            "Reasoning: low<think>\n",
            "{endpoint:?}: explicit effort must activate thinking and retain its original value",
        );
    }
}

#[tokio::test]
async fn conflicting_explicit_controls_are_rejected_before_inference() {
    for (effort, kwargs) in [
        ("low", json!({"reasoning_effort": "high"})),
        ("low", json!({"enable_thinking": false})),
        ("none", json!({"enable_thinking": true})),
    ] {
        for stream in [false, true] {
            let (router, engine) = router_with_capturing_llm_and_template(ModelChatTemplate::new(
                EFFORT_TEMPLATE,
                "conflicting-controls-contract",
            ));
            let mut wire = Endpoint::Chat.request(Some(json!(effort)));
            wire["chat_template_kwargs"] = kwargs.clone();
            wire["stream"] = json!(stream);
            let response = post_json(router, Endpoint::Chat.path(), wire).await;
            assert_invalid_request(response, &engine).await;
        }
    }
}

#[tokio::test]
async fn equal_standard_and_extension_controls_are_compatible() {
    let template = concat!(
        "Reasoning: {{ reasoning_effort }}; ",
        "Thinking: {{ enable_thinking }}",
    );
    for (effort, enabled) in [("low", true), ("none", false)] {
        let (router, engine) = router_with_capturing_llm_and_template(ModelChatTemplate::new(
            template,
            "compatible-controls-contract",
        ));
        let mut wire = Endpoint::Chat.request(Some(json!(effort)));
        wire["chat_template_kwargs"] = json!({
            "reasoning_effort": effort,
            "enable_thinking": enabled,
        });
        let response = post_json(router, Endpoint::Chat.path(), wire).await;
        assert_success(response, Endpoint::Chat, false).await;
        assert_eq!(
            engine.last_request().prompt,
            format!("Reasoning: {effort}; Thinking: {enabled}"),
        );
    }
}

#[tokio::test]
async fn template_effort_rejection_is_a_request_error_before_inference() {
    let source = concat!(
        "{% if reasoning_effort is defined and reasoning_effort not in ['low', 'high'] %}",
        "{{ raise_exception('reasoning effort is not supported by this template') }}",
        "{% endif %}{{ messages[-1].content }}",
    );
    for endpoint in [Endpoint::Chat, Endpoint::Responses] {
        let template = ModelChatTemplate::new(source, "template-owned-effort-contract");
        assert_eq!(
            template.reasoning_effort_support,
            ReasoningEffortSupport::Unknown
        );
        let (router, engine) = router_with_capturing_llm_and_template(template.clone());
        let accepted = post_json(
            router,
            endpoint.path(),
            endpoint.request(Some(json!("low"))),
        )
        .await;
        assert_success(accepted, endpoint, false).await;
        assert_eq!(engine.last_request().prompt, "hello");

        for stream in [false, true] {
            let (router, engine) = router_with_capturing_llm_and_template(template.clone());
            let mut wire = endpoint.request(Some(json!("max")));
            wire["stream"] = json!(stream);
            let rejected = post_json(router, endpoint.path(), wire).await;
            assert_invalid_request(rejected, &engine).await;
        }
    }
}

#[tokio::test]
async fn malformed_template_remains_a_server_error() {
    for endpoint in [Endpoint::Chat, Endpoint::Responses] {
        let (router, engine) = router_with_capturing_llm_and_template(ModelChatTemplate::new(
            "{% if reasoning_effort %}unclosed template",
            "malformed-template-contract",
        ));
        let response = post_json(
            router,
            endpoint.path(),
            endpoint.request(Some(json!("low"))),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::INTERNAL_SERVER_ERROR);
        let body = response_json(response).await;
        assert_eq!(body["error"]["type"], "internal_server_error");
        assert!(!engine.has_captured_request());
    }
}

#[tokio::test]
async fn unknown_or_nonthinking_templates_remain_compatible() {
    let requires_system_message = concat!(
        "{% if messages[0].role != 'system' %}",
        "{{ raise_exception('a system message is required') }}{% endif %}",
        "{{ messages[0].content }}|{{ messages[1].content }}",
    );
    for (source, expected) in [
        ("{{ messages[-1].content }}", "hello"),
        (requires_system_message, "Be concise.|hello"),
    ] {
        for endpoint in [Endpoint::Chat, Endpoint::Responses] {
            let template = ModelChatTemplate::new(source, "conversation-contract");
            if source == requires_system_message {
                assert_eq!(
                    template.reasoning_protocol,
                    ferrum_types::ModelReasoningProtocol::Unknown,
                    "the synthetic probe cannot render a template requiring a system message",
                );
            }
            let (router, engine) = router_with_capturing_llm_and_template(template);
            for effort in [None, Some(json!("low")), Some(json!("none"))] {
                let mut wire = endpoint.request(effort);
                if source == requires_system_message {
                    let messages = json!([
                        {"role": "system", "content": "Be concise."},
                        {"role": "user", "content": "hello"},
                    ]);
                    match endpoint {
                        Endpoint::Chat => wire["messages"] = messages,
                        Endpoint::Responses => wire["input"] = messages,
                    }
                }
                let response = post_json(router.clone(), endpoint.path(), wire.clone()).await;
                assert_eq!(response.status(), AxumStatusCode::OK, "{wire}");
                assert_eq!(engine.last_request().prompt, expected, "{wire}");
            }
        }
    }
}
