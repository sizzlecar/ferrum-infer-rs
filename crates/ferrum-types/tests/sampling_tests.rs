use ferrum_types::*;

#[test]
fn sampling_params_defaults_and_greedy() {
    let d = SamplingParams::default();
    assert_eq!(d.temperature, 1.0);
    assert!(d.top_p <= 1.0 && d.top_p > 0.0);

    let g = SamplingParams::greedy();
    assert_eq!(g.temperature, 0.0);
    assert!(g.top_k.is_none());
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
