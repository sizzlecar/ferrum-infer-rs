use super::*;
use ferrum_types::*;

fn template(output: AutomaticCostProbeOutput) -> AutomaticCostProbeTemplate {
    let mut request = InferenceRequest::new("actual prompt", ModelId::new("fixture"));
    request.stream = true;
    request.sampling_params = SamplingParams {
        temperature: 0.7,
        top_k: Some(3),
        repetition_penalty: 1.2,
        stop_sequences: vec!["stop".into()],
        ..Default::default()
    };
    request.metadata.insert(
        "ferrum_initial_forbidden_token_texts".into(),
        serde_json::json!(["control"]),
    );
    match output {
        AutomaticCostProbeOutput::CliText => {}
        AutomaticCostProbeOutput::ApiCompletionSse => {
            request.api_request = Some(ApiRequest::Completion(ApiCompletionRequest {
                prompt: request.prompt.clone(),
                response_format: None,
            }))
        }
        AutomaticCostProbeOutput::ApiChatSse { include_usage } => {
            request
                .metadata
                .insert(PROMPT_OPENED_REASONING_METADATA_KEY.into(), false.into());
            request.api_request = Some(ApiRequest::Chat(ApiChatRequest {
                messages: vec![ApiChatMessage {
                    role: ApiMessageRole::User,
                    content: "actual prompt".into(),
                    name: None,
                    tool_calls: vec![],
                    tool_call_id: None,
                    function_call: None,
                }],
                tools: vec![],
                tool_choice: None,
                tool_call_protocol: Default::default(),
                legacy_functions: vec![],
                legacy_function_call: None,
                response_format: None,
                stream_options: Some(ApiStreamOptions {
                    include_usage: Some(include_usage),
                }),
            }));
        }
    }
    AutomaticCostProbeTemplate::new(request, output).unwrap()
}

#[test]
fn automatic_cost_probe_configured_preserves_resolved_policy_and_api() {
    for output in [
        AutomaticCostProbeOutput::CliText,
        AutomaticCostProbeOutput::ApiCompletionSse,
        AutomaticCostProbeOutput::ApiChatSse {
            include_usage: false,
        },
        AutomaticCostProbeOutput::ApiChatSse {
            include_usage: true,
        },
    ] {
        let template = template(output);
        let (request, _) = template
            .instantiate(
                NonZeroUsize::new(7).unwrap(),
                91,
                SloAutomaticCostProbeSamplingPresetV1::Configured,
            )
            .unwrap();
        let mut expected = template.resolved_request().unwrap().sampling_params.clone();
        expected.max_tokens = 7;
        expected.seed = Some(91);
        assert_eq!(
            serde_json::to_value(request.sampling_params).unwrap(),
            serde_json::to_value(expected).unwrap()
        );
        assert_eq!(
            request.api_request,
            template.resolved_request().unwrap().api_request
        );
        assert_eq!(
            request.metadata,
            template.resolved_request().unwrap().metadata
        );
        assert_ne!(request.id, template.resolved_request().unwrap().id);
    }
}

#[test]
fn automatic_cost_probe_greedy_length_is_a_distinct_declared_supported_workload() {
    let t = template(AutomaticCostProbeOutput::CliText);
    let (r, _) = t
        .instantiate(
            NonZeroUsize::new(8).unwrap(),
            2,
            SloAutomaticCostProbeSamplingPresetV1::GreedyLength,
        )
        .unwrap();
    assert_eq!(r.sampling_params.temperature, 0.0);
    assert_eq!(r.sampling_params.repetition_penalty, 1.0);
    assert!(r.sampling_params.stop_sequences.is_empty());
    assert_eq!(r.metadata["ferrum_ignore_eos"], true);
    assert_eq!(
        r.metadata["ferrum_initial_forbidden_token_texts"],
        t.resolved_request().unwrap().metadata["ferrum_initial_forbidden_token_texts"]
    );
    assert_eq!(
        r.sampling_params.model_output_protocol,
        t.resolved_request()
            .unwrap()
            .sampling_params
            .model_output_protocol
    );
    let completion = template(AutomaticCostProbeOutput::ApiCompletionSse);
    assert!(!completion.supports_preset(SloAutomaticCostProbeSamplingPresetV1::GreedyLength));
    assert!(completion
        .instantiate(
            NonZeroUsize::new(8).unwrap(),
            2,
            SloAutomaticCostProbeSamplingPresetV1::GreedyLength
        )
        .is_err());
}

#[test]
fn automatic_cost_probe_rejects_codec_or_renderer_mismatch() {
    let cli = template(AutomaticCostProbeOutput::CliText);
    assert!(AutomaticCostProbeTemplate::new(
        cli.resolved_request().unwrap(),
        AutomaticCostProbeOutput::ApiChatSse {
            include_usage: false
        }
    )
    .is_err());
    let chat = template(AutomaticCostProbeOutput::ApiChatSse {
        include_usage: true,
    });
    assert!(AutomaticCostProbeTemplate::new(
        chat.resolved_request().unwrap(),
        AutomaticCostProbeOutput::ApiChatSse {
            include_usage: false
        }
    )
    .is_err());
    let mut missing = chat.resolved_request().unwrap();
    missing
        .metadata
        .remove(PROMPT_OPENED_REASONING_METADATA_KEY);
    assert!(AutomaticCostProbeTemplate::new(
        missing,
        AutomaticCostProbeOutput::ApiChatSse {
            include_usage: true
        }
    )
    .is_err());
}

#[tokio::test]
async fn automatic_cost_probe_real_output_plan_uses_selected_api_codec() {
    use ferrum_interfaces::output_flow::{OutputCodecKind, RequestOutputPlan};
    let raw = tokenizers::Tokenizer::new(
        tokenizers::models::wordlevel::WordLevel::builder()
            .vocab([("a".into(), 0), ("unk".into(), 1)].into_iter().collect())
            .unk_token("unk".into())
            .build()
            .unwrap(),
    );
    let mut raw = raw;
    raw.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
    let tokenizer = ferrum_tokenizer::implementations::HuggingFaceTokenizer::new(raw)
        .await
        .unwrap();
    for output in [
        AutomaticCostProbeOutput::CliText,
        AutomaticCostProbeOutput::ApiCompletionSse,
        AutomaticCostProbeOutput::ApiChatSse {
            include_usage: false,
        },
        AutomaticCostProbeOutput::ApiChatSse {
            include_usage: true,
        },
    ] {
        let (request, contract) = template(output)
            .instantiate(
                NonZeroUsize::new(7).unwrap(),
                1,
                SloAutomaticCostProbeSamplingPresetV1::Configured,
            )
            .unwrap();
        let plan = RequestOutputPlan::derive(contract, &tokenizer, &request, 2).unwrap();
        let kind = plan.codec_descriptor().unwrap().kind;
        match output {
            AutomaticCostProbeOutput::CliText => assert!(matches!(kind, OutputCodecKind::CliText)),
            AutomaticCostProbeOutput::ApiCompletionSse => assert!(matches!(
                kind,
                OutputCodecKind::CompletionsSse {
                    include_usage: true
                }
            )),
            AutomaticCostProbeOutput::ApiChatSse { include_usage } => assert!(
                matches!(kind,OutputCodecKind::ChatSse{include_usage:actual,prompt_opened:false} if actual==include_usage)
            ),
        }
    }
}

#[test]
fn automatic_cost_probe_clones_share_only_actual_owned_template_bytes() {
    let original = template(AutomaticCostProbeOutput::ApiChatSse {
        include_usage: true,
    })
    .with_response_model("public-model-alias".into());
    let clone = original.clone();
    assert!(Arc::ptr_eq(&original.request_bytes, &clone.request_bytes));
    assert!(Arc::ptr_eq(&original.prompt, &clone.prompt));
    assert!(Arc::ptr_eq(&original.response_model, &clone.response_model));
    let retained = original.retained_payload_bytes().unwrap();
    assert!(
        retained
            >= original.serialized_request().len()
                + original.prompt().len()
                + original.response_model().len()
    );
    let expected = original.resolved_request().unwrap();
    drop(original);
    let (actual, _) = clone
        .instantiate(
            NonZeroUsize::new(9).unwrap(),
            17,
            SloAutomaticCostProbeSamplingPresetV1::Configured,
        )
        .unwrap();
    assert_eq!(actual.prompt, expected.prompt);
    assert_eq!(actual.api_request, expected.api_request);
    assert_eq!(clone.response_model(), "public-model-alias");
    assert_eq!(clone.retained_payload_bytes().unwrap(), retained);
}
