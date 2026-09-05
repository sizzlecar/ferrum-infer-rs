use ferrum_types::*;

#[test]
fn sampling_params_defaults_and_greedy() {
    let d = SamplingParams::default();
    assert_eq!(d.temperature, 1.0);
    assert!(d.top_p <= 1.0 && d.top_p > 0.0);
    assert_eq!(d.model_output_protocol, ModelOutputProtocol::Text);

    let g = SamplingParams::greedy();
    assert_eq!(g.temperature, 0.0);
    assert!(g.top_k.is_none());
    assert_eq!(g.model_output_protocol, ModelOutputProtocol::Text);
}

#[test]
fn harmony_output_protocol_has_exact_typed_control_tokens() {
    assert_eq!(
        ModelOutputProtocol::HarmonyGptOss.generated_control_token_texts(),
        &[
            "<|channel|>",
            "<|message|>",
            "<|start|>",
            "<|end|>",
            "<|constrain|>",
        ]
    );
    assert_eq!(
        ModelOutputProtocol::HarmonyGptOss.preserved_special_token_texts(),
        &[
            "<|channel|>",
            "<|message|>",
            "<|start|>",
            "<|end|>",
            "<|constrain|>",
            "<|call|>",
            "<|return|>",
        ]
    );
    assert!(ModelOutputProtocol::Text
        .generated_control_token_texts()
        .is_empty());
    assert!(ModelOutputProtocol::Text
        .preserved_special_token_texts()
        .is_empty());
}

#[test]
fn model_output_protocol_is_backward_compatible_and_round_trips() {
    let mut legacy = serde_json::to_value(SamplingParams::default()).unwrap();
    legacy
        .as_object_mut()
        .unwrap()
        .remove("model_output_protocol");
    let decoded: SamplingParams = serde_json::from_value(legacy).unwrap();
    assert_eq!(decoded.model_output_protocol, ModelOutputProtocol::Text);

    let mut params = SamplingParams::default();
    params.model_output_protocol = ModelOutputProtocol::HarmonyGptOss;
    let encoded = serde_json::to_value(&params).unwrap();
    assert_eq!(encoded["model_output_protocol"], "harmony_gpt_oss");
    let decoded: SamplingParams = serde_json::from_value(encoded).unwrap();
    assert_eq!(
        decoded.model_output_protocol,
        ModelOutputProtocol::HarmonyGptOss
    );
}

#[test]
fn gemma_thought_protocol_preserves_only_native_channel_markers() {
    let protocol = ModelOutputProtocol::GemmaThought;
    assert_eq!(
        protocol.generated_control_token_texts(),
        &["<|channel>", "<channel|>"]
    );
    assert_eq!(
        protocol.preserved_special_token_texts(),
        &["<|channel>", "<channel|>"]
    );
    let encoded = serde_json::to_value(protocol).unwrap();
    assert_eq!(encoded, "gemma_thought");
    assert_eq!(
        serde_json::from_value::<ModelOutputProtocol>(encoded).unwrap(),
        protocol
    );
}

#[test]
fn reasoning_envelope_round_trips_and_requires_complete_boundaries() {
    for allow_reasoning in [false, true] {
        let mut params = SamplingParams {
            response_format: ResponseFormat::JsonObject,
            model_output_protocol: ModelOutputProtocol::GemmaThought,
            structured_output_start: StructuredOutputStart::AfterReasoningEnvelope {
                opening: "<|channel>thought\n".to_string(),
                closing: "<channel|>".to_string(),
                allow_reasoning,
            },
            ..SamplingParams::default()
        };
        params.validate().unwrap();
        let encoded = serde_json::to_value(&params).unwrap();
        assert_eq!(
            encoded["structured_output_start"]["mode"],
            "after_reasoning_envelope"
        );
        let decoded: SamplingParams = serde_json::from_value(encoded).unwrap();
        assert_eq!(
            decoded.structured_output_start,
            params.structured_output_start
        );
        decoded.validate().unwrap();
        for (opening, closing) in [("", "<channel|>"), ("<|channel>thought\n", "")] {
            params.structured_output_start = StructuredOutputStart::AfterReasoningEnvelope {
                opening: opening.to_string(),
                closing: closing.to_string(),
                allow_reasoning,
            };
            assert!(params.validate().is_err());
        }
    }
}

#[test]
fn harmony_structured_boundary_requires_its_protocol_and_round_trips() {
    let mut params = SamplingParams {
        response_format: ResponseFormat::JsonObject,
        structured_output_start: StructuredOutputStart::HarmonyFinal,
        ..SamplingParams::default()
    };
    assert!(params.validate().is_err());
    params.model_output_protocol = ModelOutputProtocol::HarmonyGptOss;
    params.validate().unwrap();
    let encoded = serde_json::to_value(&params).unwrap();
    assert_eq!(
        encoded["structured_output_start"],
        serde_json::json!({"mode": "harmony_final"})
    );
    let decoded: SamplingParams = serde_json::from_value(encoded).unwrap();
    assert_eq!(
        decoded.structured_output_start,
        StructuredOutputStart::HarmonyFinal
    );
    decoded.validate().unwrap();

    let mut legacy = serde_json::to_value(SamplingParams::default()).unwrap();
    legacy
        .as_object_mut()
        .unwrap()
        .remove("structured_output_start");
    let decoded: SamplingParams = serde_json::from_value(legacy).unwrap();
    assert_eq!(
        decoded.structured_output_start,
        StructuredOutputStart::Immediate
    );
}

#[test]
fn sampling_params_validate_checks() {
    let mut p = SamplingParams::default();
    p.temperature = -0.1;
    assert!(p.validate().is_err());

    let mut p = SamplingParams::default();
    p.top_p = 0.0; // invalid
    assert!(p.validate().is_err());

    let mut p = SamplingParams::default();
    p.top_k = Some(0);
    assert!(p.validate().is_err());

    let mut p = SamplingParams::default();
    p.min_p = Some(1.1);
    assert!(p.validate().is_err());

    let mut p = SamplingParams::default();
    p.response_completion_boundary = ResponseCompletionBoundary::AfterDelimiterAndPayload {
        delimiter: String::new(),
        alternate_envelope: None,
    };
    assert!(p.validate().is_err());

    let mut p = SamplingParams::default();
    p.response_completion_boundary = ResponseCompletionBoundary::AfterDelimiterAndPayload {
        delimiter: "</think>".to_string(),
        alternate_envelope: Some(ResponseCompletionEnvelope {
            open_token_text: "<tool_call>".to_string(),
            close_token_text: "</tool_call>".to_string(),
            max_envelopes: 0,
        }),
    };
    assert!(p.validate().is_err());
}

#[test]
fn response_completion_boundary_is_backward_compatible_and_round_trips() {
    let mut legacy = serde_json::to_value(SamplingParams::default()).unwrap();
    legacy
        .as_object_mut()
        .unwrap()
        .remove("response_completion_boundary");
    let decoded: SamplingParams = serde_json::from_value(legacy).unwrap();
    assert_eq!(
        decoded.response_completion_boundary,
        ResponseCompletionBoundary::Immediate
    );

    let mut legacy_boundary = serde_json::to_value(SamplingParams::default()).unwrap();
    legacy_boundary.as_object_mut().unwrap().insert(
        "response_completion_boundary".to_string(),
        serde_json::json!({
            "mode": "after_delimiter_and_payload",
            "delimiter": "</think>"
        }),
    );
    let decoded: SamplingParams = serde_json::from_value(legacy_boundary).unwrap();
    assert_eq!(
        decoded.response_completion_boundary,
        ResponseCompletionBoundary::AfterDelimiterAndPayload {
            delimiter: "</think>".to_string(),
            alternate_envelope: None,
        }
    );

    let mut params = SamplingParams::default();
    params.response_completion_boundary = ResponseCompletionBoundary::AfterDelimiterAndPayload {
        delimiter: "</think>".to_string(),
        alternate_envelope: Some(ResponseCompletionEnvelope {
            open_token_text: "<tool_call>".to_string(),
            close_token_text: "</tool_call>".to_string(),
            max_envelopes: 32,
        }),
    };
    let decoded: SamplingParams =
        serde_json::from_value(serde_json::to_value(params).unwrap()).unwrap();
    assert_eq!(
        decoded.response_completion_boundary,
        ResponseCompletionBoundary::AfterDelimiterAndPayload {
            delimiter: "</think>".to_string(),
            alternate_envelope: Some(ResponseCompletionEnvelope {
                open_token_text: "<tool_call>".to_string(),
                close_token_text: "</tool_call>".to_string(),
                max_envelopes: 32,
            }),
        }
    );
}

#[test]
fn special_tokens_default() {
    let st = SpecialTokens::default();
    assert!(st.eos_token.is_none());
}
