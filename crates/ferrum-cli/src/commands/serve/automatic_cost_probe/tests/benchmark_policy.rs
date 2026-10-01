//! Converter half of the policy proof. These requests come from the real
//! benchmark body builder and public server converter, without inference.
use super::*;
use ferrum_bench_core::env::HttpRequestSampling;
use ferrum_types::{InferenceEvidenceRequest, ModelOutputProtocol, ResponseCompletionBoundary};
use std::num::NonZeroUsize;

#[test]
fn benchmark_chat_converter_matches_only_usage_enabled_greedy_length_template() {
    let config = automatic_config();
    let model = ModelChatTemplate::new(
        include_str!("../../../../../../ferrum-server/tests/fixtures/chat_template/Qwen__Qwen3.5-35B-A3B/template.jinja"),
        "thinking-capable policy fixture",
    );
    assert!(model.supports_thinking_control());
    let body = crate::commands::chat_request::chat_completion_body(
        "public-alias",
        USER_TEXT,
        547,
        true,
        Some(false),
        None,
        HttpRequestSampling {
            temperature: 0.0,
            top_k: None,
            top_p: Some(1.0),
            repetition_penalty: Some(1.0),
            seed: Some(42),
        },
    );
    assert!(body.get("top_k").is_none(), "omitted on the actual wire");
    assert_eq!(body["ignore_eos"], true);
    assert_eq!(body["chat_template_kwargs"]["enable_thinking"], false);
    let http: ChatCompletionsRequest = serde_json::from_value(body).unwrap();
    assert_eq!(
        http.stream_options.as_ref().unwrap().include_usage,
        Some(true)
    );
    let benchmark = prepare_model_chat_request(
        &http,
        config.model.model_id.as_str(),
        &model,
        Some(false),
        true,
    )
    .unwrap();
    assert!(benchmark.stream);
    assert_eq!(benchmark.sampling_params.top_k, None);
    assert!(benchmark.sampling_params.stop_sequences.is_empty());
    assert_eq!(benchmark.metadata["ferrum_ignore_eos"], true);
    assert_eq!(
        benchmark.metadata[ferrum_types::PROMPT_OPENED_REASONING_METADATA_KEY],
        false
    );
    assert_eq!(
        benchmark.sampling_params.model_output_protocol,
        ModelOutputProtocol::Text
    );
    assert_eq!(
        benchmark.sampling_params.response_completion_boundary,
        ResponseCompletionBoundary::Immediate
    );
    assert_eq!(
        benchmark.evidence_request,
        InferenceEvidenceRequest::default(),
        "profile-detail off adds no evidence opt-in"
    );

    for include_usage in [false, true] {
        // Server --disable-thinking supplies the same typed default as the
        // client override. The probe uses its real renderer and preset path.
        let template = chat_template(
            &config,
            &model,
            "public-alias",
            547,
            include_usage,
            Some(false),
            true,
            None,
        )
        .unwrap();
        assert_eq!(
            template.output(),
            AutomaticCostProbeOutput::ApiChatSse { include_usage }
        );
        let (probe, _checked_contract) = template
            .instantiate(
                NonZeroUsize::new(547).unwrap(),
                42,
                SloAutomaticCostProbeSamplingPresetV1::GreedyLength,
            )
            .unwrap();
        assert_eq!(probe.prompt, benchmark.prompt);
        assert_eq!(probe.model_id, benchmark.model_id);
        assert_eq!(probe.stream, benchmark.stream);
        assert_eq!(
            probe.metadata, benchmark.metadata,
            "includes real renderer masks and reasoning state"
        );
        assert_eq!(probe.evidence_request, benchmark.evidence_request);
        assert_eq!(
            serde_json::to_value(&probe.sampling_params).unwrap(),
            serde_json::to_value(&benchmark.sampling_params).unwrap()
        );
        assert_eq!(probe.api_request == benchmark.api_request, include_usage);
        let Some(ApiRequest::Chat(mut api)) = probe.api_request else {
            panic!("real Chat API")
        };
        assert_eq!(
            api.stream_options.as_ref().unwrap().include_usage,
            Some(include_usage)
        );
        api.stream_options.as_mut().unwrap().include_usage = Some(true);
        assert_eq!(
            Some(ApiRequest::Chat(api)),
            benchmark.api_request,
            "usage is the only API policy difference"
        );

        let (short, _) = template
            .instantiate(
                NonZeroUsize::new(3).unwrap(),
                0,
                SloAutomaticCostProbeSamplingPresetV1::GreedyLength,
            )
            .unwrap();
        let mut normalized = short.sampling_params.clone();
        assert_eq!(normalized.max_tokens, 3);
        assert_eq!(normalized.seed, Some(0));
        normalized.max_tokens = benchmark.sampling_params.max_tokens;
        normalized.seed = benchmark.sampling_params.seed;
        assert_eq!(
            serde_json::to_value(normalized).unwrap(),
            serde_json::to_value(&benchmark.sampling_params).unwrap()
        );
        assert_eq!(short.metadata, benchmark.metadata);
    }
}
