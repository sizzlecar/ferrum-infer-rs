//! Discovery must describe actual controls without inventing effort levels.
use super::*;
use ferrum_types::{ReasoningEffort, ReasoningEffortSupport};
use std::collections::BTreeSet;

fn toggle_template() -> ModelChatTemplate {
    ModelChatTemplate::new(
        "{{ messages[-1].content }}{% if enable_thinking | default(true) %}<think>{% else %}<think></think>{% endif %}",
        "toggle-contract",
    )
}

async fn model_entries(server: AxumServer) -> Vec<Value> {
    let response = get(server.build_router(), "/v1/models").await;
    assert_eq!(response.status(), AxumStatusCode::OK);
    response_json(response).await["data"]
        .as_array()
        .unwrap()
        .clone()
}

#[tokio::test]
async fn declared_efforts_preserve_unknown_and_explicitly_empty_support() {
    for declaration in [
        ReasoningEffortSupport::Unknown,
        ReasoningEffortSupport::Declared(BTreeSet::new()),
        ReasoningEffortSupport::Declared(BTreeSet::from([
            ReasoningEffort::Low,
            ReasoningEffort::High,
        ])),
    ] {
        let mut template = ModelChatTemplate::new(
            "{{ reasoning_effort | default('medium') }} {{ messages[-1].content }}",
            "effort-contract",
        );
        template.reasoning_effort_support = declaration.clone();
        let entries = model_entries(
            AxumServer::from_llm(Arc::new(StubLlm::new("ok"))).with_prompt_template(Some(template)),
        )
        .await;
        match declaration {
            ReasoningEffortSupport::Unknown => assert!(entries[0].get("reasoning").is_none()),
            ReasoningEffortSupport::Declared(efforts) => {
                assert_eq!(
                    entries[0]["reasoning"],
                    json!({"supported_efforts": efforts})
                );
            }
        }
    }
}

#[tokio::test]
async fn harmony_effort_declaration_does_not_imply_a_template_switch() {
    let mut template = toggle_template();
    template.set_output_protocol(ferrum_types::ModelOutputProtocol::HarmonyGptOss);
    template.reasoning_effort_support = ReasoningEffortSupport::Declared(BTreeSet::from([
        ReasoningEffort::Low,
        ReasoningEffort::Medium,
        ReasoningEffort::High,
    ]));
    assert_eq!(template.reasoning_protocol, ModelReasoningProtocol::None);
    assert_eq!(
        template.reasoning_capability(),
        ModelReasoningProtocol::ModelGenerated
    );
    let entries = model_entries(
        AxumServer::from_llm(Arc::new(StubLlm::new("ok"))).with_prompt_template(Some(template)),
    )
    .await;
    assert_eq!(
        entries[0]["reasoning"],
        json!({"supported_efforts": ["low", "medium", "high"]})
    );
}

#[tokio::test]
async fn thinking_metadata_uses_effective_default_for_loaded_aliases_and_adapters() {
    for override_enabled in [None, Some(false), Some(true)] {
        let registry = ServedModelRegistry::try_new(
            "stub-model",
            ServedModelKind::Llm,
            vec!["public-model".to_owned(), "second-alias".to_owned()],
            vec![LoraAdapterModel::new(
                "sql",
                "public-model:sql",
                "/tmp/adapter",
            )],
        )
        .unwrap();
        let entries = model_entries(
            AxumServer::from_llm(Arc::new(StubLlm::new("ok")))
                .with_served_model_registry(registry)
                .with_prompt_template(Some(toggle_template()))
                .with_default_enable_thinking(override_enabled),
        )
        .await;
        assert_eq!(entries.len(), 3);
        for entry in entries {
            assert_eq!(
                entry["reasoning"],
                json!({
                    "thinking": {"default_enabled": override_enabled.unwrap_or(true)}
                })
            );
        }
    }
}

#[tokio::test]
async fn template_and_model_generated_defaults_are_preserved() {
    for (source, default_enabled, protocol) in [
        (
            "{% if enable_thinking | default(false) %}<think>{% else %}<think></think>{% endif %}",
            false,
            ModelReasoningProtocol::PromptOpened,
        ),
        (
            "{% if not (enable_thinking | default(true)) %}<think></think>{% endif %}",
            true,
            ModelReasoningProtocol::ModelGenerated,
        ),
    ] {
        let template = ModelChatTemplate::new(source, "probe-contract");
        assert_eq!(template.reasoning_protocol, protocol);
        assert_eq!(template.reasoning_default_enabled, default_enabled);
        let entries = model_entries(
            AxumServer::from_llm(Arc::new(StubLlm::new("ok"))).with_prompt_template(Some(template)),
        )
        .await;
        assert_eq!(
            entries[0]["reasoning"],
            json!({
                "thinking": {"default_enabled": default_enabled}
            })
        );
    }
}

#[tokio::test]
async fn qwen35_template_advertises_toggle_without_inventing_strength_levels() {
    let template = ModelChatTemplate::new(
        include_str!("../../../tests/fixtures/chat_template/Qwen__Qwen3.5-35B-A3B/template.jinja"),
        "arbitrary-source-name",
    );
    let entries = model_entries(
        AxumServer::from_llm(Arc::new(StubLlm::new("ok"))).with_prompt_template(Some(template)),
    )
    .await;
    assert_eq!(
        entries[0]["reasoning"],
        json!({
            "thinking": {"default_enabled": true}
        })
    );
}

#[tokio::test]
async fn output_reasoning_alone_does_not_advertise_a_switch() {
    for source in [
        "{{ messages[-1].content }}<think>",
        "{% if not (enable_thinking | default(true)) %}<think></think><think>{% endif %}",
        "{% if enable_thinking | default(true) %}<think>{% else %}{{ raise_exception('cannot disable') }}{% endif %}",
        "{{ messages[-1].content }}",
    ] {
        let entries = model_entries(
            AxumServer::from_llm(Arc::new(StubLlm::new("ok")))
                .with_prompt_template(Some(ModelChatTemplate::new(source, "qwen3.5"))),
        ).await;
        assert!(entries[0].get("reasoning").is_none());
    }
}

#[tokio::test]
async fn unrelated_or_unloaded_registry_entries_do_not_inherit_llm_controls() {
    for (kind, engine_id, loaded) in [
        (ServedModelKind::Embedding, "stub-model", true),
        (ServedModelKind::Llm, "other-model", true),
        (ServedModelKind::Llm, "stub-model", false),
    ] {
        let mut state = AppState::default();
        if loaded {
            state = state.with_llm(Arc::new(StubLlm::new("ok")));
        }
        state = state.with_prompt_template(Some(toggle_template()));
        state.served_model_registry = Arc::new(
            ServedModelRegistry::try_new(engine_id, kind, vec!["public-model".to_owned()], vec![])
                .unwrap(),
        );
        let entries = model_entries(AxumServer::from_state(state)).await;
        assert!(entries[0].get("reasoning").is_none());
    }
}

#[tokio::test]
async fn advertised_switch_reaches_sync_and_stream_template_rendering() {
    for stream in [false, true] {
        for enabled in [false, true] {
            let (router, engine) = router_with_capturing_llm_and_template(toggle_template());
            let metadata = response_json(get(router.clone(), "/v1/models").await).await;
            assert!(metadata["data"][0]["reasoning"]["thinking"].is_object());
            let response = post_json(
                router,
                "/v1/chat/completions",
                json!({
                    "model": "served-alias",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": stream,
                    "chat_template_kwargs": {"enable_thinking": enabled}
                }),
            )
            .await;
            assert_eq!(response.status(), AxumStatusCode::OK);
            let body = response_text(response).await;
            assert!(!body.is_empty());
            assert_eq!(
                engine.last_request().prompt,
                if enabled {
                    "hello<think>"
                } else {
                    "hello<think></think>"
                }
            );
        }
    }
}
